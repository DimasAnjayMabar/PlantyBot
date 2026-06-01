// lib/chats/services/chat_service.dart
import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:frontend/services/token_storage.dart';
import 'package:http/http.dart' as http;
import 'package:audioplayers/audioplayers.dart';

// ---------------------------------------------------------------------------
// Konfigurasi
// ---------------------------------------------------------------------------

final _dio = Dio(
  BaseOptions(
    baseUrl: 'http://localhost:8000',
    connectTimeout: const Duration(seconds: 30),
    receiveTimeout: const Duration(seconds: 60),
    headers: {'Content-Type': 'application/json'},
  ),
);

const _kBaseUrl = 'http://localhost:8000';
const _kTokenRefreshInterval = Duration(minutes: 25);

// ---------------------------------------------------------------------------
// Model Preference Storage - Menyimpan model yang dipilih user
// ---------------------------------------------------------------------------

class ModelPreferenceStorage {
  static const String _keySelectedModel = 'selected_model_id';
  static const String _keySelectedMode = 'selected_llm_mode'; // 'groq' or 'local'
  
  /// Simpan model yang dipilih
  static Future<void> saveSelectedModel(String modelId) async {
    await TokenStorage.write(key: _keySelectedModel, value: modelId);
    print('💾 Model preference saved: $modelId');
  }
  
  /// Ambil model yang tersimpan
  static Future<String?> getSelectedModel() async {
    return await TokenStorage.read(key: _keySelectedModel);
  }
  
  /// Simpan mode LLM (groq/local)
  static Future<void> saveSelectedMode(String mode) async {
    await TokenStorage.write(key: _keySelectedMode, value: mode);
    print('💾 Mode preference saved: $mode');
  }
  
  /// Ambil mode yang tersimpan
  static Future<String?> getSelectedMode() async {
    return await TokenStorage.read(key: _keySelectedMode);
  }
  
  /// Hapus semua preference model
  static Future<void> clearModelPreference() async {
    await TokenStorage.delete(key: _keySelectedModel);
    await TokenStorage.delete(key: _keySelectedMode);
  }
}

// ---------------------------------------------------------------------------
// Token Storage — Abstraction Layer
//
// Web/Chrome  → SharedPreferences (localStorage) — persisten lintas restart browser
// Android     → FlutterSecureStorage (EncryptedSharedPreferences) — enkripsi AES-256
//               kunci disimpan di Android Keystore (hardware-backed)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

class ChatTopic {
  final int id;
  String title;
  final String createdAt;

  ChatTopic({required this.id, required this.title, required this.createdAt});

  factory ChatTopic.fromJson(Map<String, dynamic> j) => ChatTopic(
        id: j['id'] as int,
        title: j['title'] as String,
        createdAt: j['created_at'] as String,
      );
}

class ChatMessage {
  final int id;
  final int chatId;
  final String question;
  String response;
  String processingStatus;
  final String createdAt;

  ChatMessage({
    required this.id,
    required this.chatId,
    required this.question,
    required this.response,
    required this.processingStatus,
    required this.createdAt,
  });

  ChatMessage copyWith({
    String? response,
    String? processingStatus,
  }) {
    return ChatMessage(
      id: id,
      chatId: chatId,
      question: question,
      response: response ?? this.response,
      processingStatus: processingStatus ?? this.processingStatus,
      createdAt: createdAt,
    );
  }

  factory ChatMessage.pending({
    required int id,
    required int chatId,
    required String question,
    required String createdAt,
  }) =>
      ChatMessage(
        id: id,
        chatId: chatId,
        question: question,
        response: '',
        processingStatus: 'pending',
        createdAt: createdAt,
      );

  factory ChatMessage.fromJson(Map<String, dynamic> j) => ChatMessage(
        id: j['id'] as int,
        chatId: j['chat_id'] as int,
        question: j['question'] as String,
        response: j['response'] as String? ?? '',
        processingStatus: j['processing_status'] as String? ?? 'pending',
        createdAt: j['created_at'] as String,
      );

  bool get isPending => processingStatus == 'pending';
  bool get isDone => processingStatus == 'done';
  bool get isFailed => processingStatus == 'failed';
  bool get isDisconnected => processingStatus == 'disconnected';
  bool get isStopped => processingStatus == 'stopped';
}

class ChatUserProfile {
  final String name;
  final String email;
  final String username;

  const ChatUserProfile({
    required this.name,
    required this.email,
    required this.username,
  });

  factory ChatUserProfile.fromJson(Map<String, dynamic> j) => ChatUserProfile(
        name: j['name'] as String,
        email: j['email'] as String,
        username: j['username'] as String,
      );
}

// ---------------------------------------------------------------------------
// SSE Client
// ---------------------------------------------------------------------------

class SseEvent {
  final String type;
  final String data;
  const SseEvent({required this.type, required this.data});
}

class SseClient {
  static Stream<SseEvent> subscribe(String url, String token) async* {
    final client = http.Client();
    final request = http.Request('GET', Uri.parse(url))
      ..headers['Accept'] = 'text/event-stream'
      ..headers['Cache-Control'] = 'no-cache'
      ..headers['Authorization'] = 'Bearer $token'
      ..headers['Connection'] = 'keep-alive';

    http.StreamedResponse response;
    try {
      response = await client.send(request);
      print('✅ SSE Connected, status: ${response.statusCode}');
    } catch (e) {
      print('❌ SSE Connection failed: $e');
      throw Exception('SSE connect gagal: $e');
    }

    if (response.statusCode != 200) {
      print('❌ SSE HTTP error: ${response.statusCode}');
      throw Exception('SSE connect gagal: HTTP ${response.statusCode}');
    }

    String buffer = '';
    String currentEventType = 'message';

    try {
      await for (final chunk in response.stream.transform(utf8.decoder)) {
        buffer += chunk;

        final lines = buffer.split('\n');
        buffer = lines.last;

        for (var i = 0; i < lines.length - 1; i++) {
          final line = lines[i].trim();

          if (line.isEmpty) {
            continue;
          }

          if (line.startsWith('event:')) {
            currentEventType = line.substring(6).trim();
            print('📡 SSE Event Type: $currentEventType');
          } else if (line.startsWith('data:')) {
            final data = line.substring(5).trim();
            print('📡 SSE Data: $data');
            if (data.isNotEmpty) {
              yield SseEvent(type: currentEventType, data: data);
              currentEventType = 'message';
            }
          } else if (line.startsWith(':')) {
            print('💓 SSE Heartbeat');
          }
        }
      }
    } finally {
      client.close();
      print('🏁 SSE Client closed');
    }
  }
}

// ---------------------------------------------------------------------------
// Chat Service (Main API Logic)
// ---------------------------------------------------------------------------

class ChatService {
  String? _accessToken;
  int? _userId;
  Timer? _tokenTimer;
  
  // ── Model State ──────────────────────────────────────────────────────────
  String? _currentModelId;
  String? _currentMode; // 'groq' or 'local'

  // Instance AudioPlayer & State Tracker untuk memutar TTS
  final AudioPlayer _audioPlayer = AudioPlayer();
  final ValueNotifier<int?> playingTtsId = ValueNotifier<int?>(null);
  int? _currentTtsRequestDetailId;

  String? get accessToken => _accessToken;
  int? get userId => _userId;
  String? get currentModelId => _currentModelId;

  Map<String, dynamic> get _authHeader => {'Authorization': 'Bearer $_accessToken'};

  // Callbacks untuk UI
  final VoidCallback? onForceLogout;
  final void Function(String? token)? onTokenUpdated;
  final void Function(String? modelId)? onModelChanged; // ← callback baru

  ChatService({
    this.onForceLogout,
    this.onTokenUpdated,
    this.onModelChanged,
  }) {
    // Memantau status player agar tombol kembali normal jika audio selesai/stop otomatis
    _audioPlayer.onPlayerComplete.listen((_) {
      playingTtsId.value = null;
    });
    _audioPlayer.onPlayerStateChanged.listen((state) {
      if (state == PlayerState.stopped || state == PlayerState.completed) {
        playingTtsId.value = null;
      }
    });
  }

  // -------------------------------------------------------------------------
  // Model Preference Methods
  // -------------------------------------------------------------------------

  /// Inisialisasi model preference dari storage
  Future<void> initModelPreference() async {
    try {
      // Ambil dari storage
      _currentModelId = await ModelPreferenceStorage.getSelectedModel();
      _currentMode = await ModelPreferenceStorage.getSelectedMode();
      
      if (_currentModelId != null && _currentMode != null) {
        print('🔄 Loading saved model preference: mode=$_currentMode, model=$_currentModelId');
        
        // Sinkronkan dengan backend
        final success = await _setModelInternal(_currentMode!, path: _currentModelId);
        if (success) {
          print('✅ Model preference synced with backend');
          onModelChanged?.call(_currentModelId);
        } else {
          print('⚠️ Failed to sync model with backend, using saved preference anyway');
        }
      } else {
        // Jika belum ada preference, gunakan default dari backend
        await _syncModelFromBackend();
      }
    } catch (e) {
      print('❌ Error loading model preference: $e');
    }
  }

  /// Sync model dari backend (untuk memastikan konsistensi)
  Future<void> _syncModelFromBackend() async {
    try {
      final res = await _dio.get(
        '/models/active',
        options: Options(headers: _authHeader),
      );
      if (res.data['success'] == true) {
        final modelId = res.data['data']['model_id'] as String;
        final mode = res.data['data']['mode'] as String;
        
        _currentModelId = modelId;
        _currentMode = mode;
        
        // Simpan ke storage
        await ModelPreferenceStorage.saveSelectedModel(modelId);
        await ModelPreferenceStorage.saveSelectedMode(mode);
        
        print('✅ Synced model from backend: mode=$mode, model=$modelId');
        onModelChanged?.call(modelId);
      }
    } catch (e) {
      print('⚠️ Failed to sync model from backend: $e');
      // Fallback ke default
      _currentModelId = 'llama-3.3-70b-versatile';
      _currentMode = 'groq';
    }
  }

  /// Internal method untuk set model tanpa menyimpan ke storage (untuk sync)
  Future<bool> _setModelInternal(String mode, {String? path}) async {
    try {
      final resp = await _dio.post(
        '/models/set-model',
        data: {'model_id': path},
        options: Options(headers: _authHeader)
      );
      return resp.data['success'] == true;
    } catch (e) {
      print('❌ Internal set model failed: $e');
      return false;
    }
  }

  /// Set model dan simpan ke storage
  Future<bool> setModel(String mode, {String? path}) async {
    try {
      final resp = await _dio.post(
        '/models/set-model',
        data: {'model_id': path},
        options: Options(headers: _authHeader)
      );
      
      if (resp.data['success'] == true) {
        // Simpan ke memory
        _currentModelId = path;
        _currentMode = mode;
        
        // Simpan ke storage
        if (path != null) {
          await ModelPreferenceStorage.saveSelectedModel(path);
        }
        await ModelPreferenceStorage.saveSelectedMode(mode);
        
        print('✅ Model changed successfully: mode=$mode, model=$path');
        onModelChanged?.call(path);
        return true;
      }
      return false;
    } catch (e) {
      print('❌ Set model failed: $e');
      return false;
    }
  }

  /// Get current active model
  Future<String?> getCurrentModel() async {
    if (_currentModelId != null) return _currentModelId;
    
    // Coba sync dari backend
    await _syncModelFromBackend();
    return _currentModelId;
  }

  /// Get current mode
  Future<String?> getCurrentMode() async {
    if (_currentMode != null) return _currentMode;
    await _syncModelFromBackend();
    return _currentMode;
  }

  // -------------------------------------------------------------------------
  // RAG Mode Management
  // -------------------------------------------------------------------------

  /// Mendapatkan mode RAG yang sedang aktif dari backend
  Future<String> getRagMode() async {
    try {
      final res = await _dio.get(
        '/rag/mode',
        options: Options(headers: _authHeader),
      );
      if (res.data['success'] == true) {
        final mode = res.data['data']['mode'] as String;
        return mode;
      }
      return 'improved';
    } catch (e) {
      print('❌ Error getting RAG mode: $e');
      return 'improved';
    }
  }

  /// Mengganti mode RAG
  Future<bool> setRagMode(String mode) async {
    try {
      final res = await _dio.post(
        '/rag/set-mode',
        data: {'mode': mode},
        options: Options(headers: _authHeader),
      );
      if (res.data['success'] == true) {
        // Gunakan RAGPreferenceStorage dari token_storage.dart
        await RAGPreferenceStorage.saveRagMode(mode);
        print('✅ RAG mode changed to: $mode');
        return true;
      }
      return false;
    } catch (e) {
      print('❌ Set RAG mode failed: $e');
      return false;
    }
  }

  /// Initialize RAG mode preference (panggil setelah login)
  Future<void> initRagMode() async {
    try {
      // Coba ambil dari storage dulu menggunakan RAGPreferenceStorage
      String? savedMode = await RAGPreferenceStorage.getSavedRagMode();
      
      if (savedMode != null) {
        // Sync ke backend
        await setRagMode(savedMode);
      } else {
        // Ambil dari backend
        final currentMode = await getRagMode();
        await RAGPreferenceStorage.saveRagMode(currentMode);
      }
    } catch (e) {
      print('⚠️ Error initializing RAG mode: $e');
    }
  }

  // -------------------------------------------------------------------------
  // Auth Methods
  // -------------------------------------------------------------------------

  /// Inisialisasi auth saat app pertama kali dibuka.
  /// Membaca token dari storage (localStorage di web, EncryptedSharedPrefs di Android).
  /// Return true jika token ditemukan dan valid — langsung masuk ke home.
  Future<bool> initAuth() async {
    try {
      _accessToken = await TokenStorage.read(key: 'access_token');
      final uid = await TokenStorage.read(key: 'user_id');
      _userId = uid != null ? int.tryParse(uid) : null;
    } catch (_) {}

    if (_accessToken == null || _accessToken!.isEmpty || _userId == null) {
      return false;
    }
    _startTokenTimer();
    return true;
  }

  /// Simpan token setelah login berhasil.
  /// Dipanggil dari auth service / login handler setelah response login diterima.
  Future<void> saveAuthData({
    required String accessToken,
    required String refreshToken,
    required int userId,
  }) async {
    await Future.wait([
      TokenStorage.write(key: 'access_token', value: accessToken),
      TokenStorage.write(key: 'refresh_token', value: refreshToken),
      TokenStorage.write(key: 'user_id', value: userId.toString()),
      TokenStorage.write(
        key: 'session_created_at',
        value: DateTime.now().toIso8601String(),
      ),
    ]);
    _accessToken = accessToken;
    _userId = userId;
    _startTokenTimer();
    
    // Setelah login, sync model preference
    await initModelPreference();
  }

  void _startTokenTimer() {
    _tokenTimer?.cancel();
    print('🔄 Starting token refresh timer with interval: ${_kTokenRefreshInterval.inMinutes} minutes');
    _tokenTimer = Timer.periodic(_kTokenRefreshInterval, (_) => _silentRefresh());
  }

  /// Silent refresh token setiap interval.
  /// Membaca refresh_token dari storage, kirim ke backend, simpan token baru.
  /// Jika refresh_token tidak ada atau expired → force logout.
  Future<void> _silentRefresh() async {
    print('🔄 [${DateTime.now()}] Running silent refresh...');
    
    final rt = await TokenStorage.read(key: 'refresh_token');
    if (rt == null || rt.isEmpty) {
      print('❌ [${DateTime.now()}] No refresh token found, forcing logout');
      _forceLogout();
      return;
    }
    
    print('📝 [${DateTime.now()}] Refresh token found, attempting to refresh...');
    
    try {
      final res = await _dio.post('/users/refresh-token', 
        data: {'refresh_token': rt},
        options: Options(
          headers: _authHeader,
          validateStatus: (status) => status! < 500,
        ),
      );
      
      if (res.statusCode == 200) {
        final d = res.data['data'] as Map<String, dynamic>;
        final newAccessToken = d['access_token'] as String;
        final newRefreshToken = d['refresh_token'] as String;
        
        await Future.wait([
          TokenStorage.write(key: 'access_token', value: newAccessToken),
          TokenStorage.write(key: 'refresh_token', value: newRefreshToken),
        ]);
        
        _accessToken = newAccessToken;
        onTokenUpdated?.call(_accessToken);
        
        print('✅ [${DateTime.now()}] Token refreshed successfully!');
      } else {
        print('❌ [${DateTime.now()}] Refresh failed with status: ${res.statusCode}');
        
        if (res.statusCode == 401 || res.statusCode == 403) {
          print('⚠️ Token expired or invalid, forcing logout');
          _forceLogout();
        }
      }
    } on DioException catch (e) {
      print('❌ [${DateTime.now()}] DioException during refresh: ${e.message}');
      
      if (e.response?.statusCode == 401 || e.response?.statusCode == 403) {
        print('⚠️ Token expired or invalid, forcing logout');
        _forceLogout();
      }
    } catch (e) {
      print('❌ [${DateTime.now()}] Unexpected error during refresh: $e');
    }
  }

  /// Force logout tanpa memanggil API — hanya bersihkan state lokal.
  /// Dipanggil saat token expired atau refresh gagal.
  void _forceLogout() {
    _tokenTimer?.cancel();
    _accessToken = null;
    _userId = null;
    _audioPlayer.stop();
    playingTtsId.value = null;
    onForceLogout?.call();
  }

  /// Force logout + hapus semua data dari storage.
  /// Dipanggil saat ada error auth yang tidak bisa di-recover.
  Future<void> forceLogout() async {
    _tokenTimer?.cancel();
    await TokenStorage.deleteAll();
    await ModelPreferenceStorage.clearModelPreference(); // ← hapus juga model preference
    _accessToken = null;
    _userId = null;
    _currentModelId = null;
    _currentMode = null;
    await _audioPlayer.stop();
    playingTtsId.value = null;
  }

  /// Logout normal — panggil API logout dulu, lalu bersihkan storage.
  Future<void> logout() async {
    try {
      await _dio.post('/users/logout', options: Options(headers: _authHeader));
    } catch (_) {}
    await forceLogout();
  }

  // -------------------------------------------------------------------------
  // API Methods
  // -------------------------------------------------------------------------

  Future<List<ChatTopic>> fetchTopics() async {
    try {
      final res = await _dio.get('/topics', options: Options(headers: _authHeader));
      final data = res.data['data'] as Map<String, dynamic>;
      return (data['topics'] as List)
          .map((e) => ChatTopic.fromJson(e as Map<String, dynamic>))
          .toList();
    } catch (_) {
      return [];
    }
  }

  Future<ChatUserProfile?> fetchProfile() async {
    if (_userId == null) return null;
    try {
      final res = await _dio.get('/users/$_userId', options: Options(headers: _authHeader));
      final data = res.data['data'] as Map<String, dynamic>;
      return ChatUserProfile.fromJson(data);
    } catch (_) {
      return null;
    }
  }

  Future<List<ChatMessage>> fetchMessages(int chatId) async {
    try {
      final res = await _dio.get('/topics/$chatId', options: Options(headers: _authHeader));
      final data = res.data['data'] as Map<String, dynamic>;
      return (data['messages'] as List)
          .map((e) => ChatMessage.fromJson(e as Map<String, dynamic>))
          .toList();
    } catch (_) {
      return [];
    }
  }

  Future<ChatMessage?> sendMessage({
    required int? chatId,
    required String question,
  }) async {
    try {
      final res = await _dio.post(
        '/chat/send',
        data: {'chat_id': chatId, 'question': question},
        options: Options(headers: _authHeader),
      );
      final data = res.data['data'] as Map<String, dynamic>;
      return ChatMessage.pending(
        id: data['id'] as int,
        chatId: data['chat_id'] as int,
        question: data['question'] as String,
        createdAt: data['created_at'] as String,
      );
    } catch (_) {
      return null;
    }
  }

  Future<ChatMessage?> editMessage(int messageId, String newQuestion) async {
    try {
      final res = await _dio.patch(
        '/chat/edit/$messageId',
        data: {'question': newQuestion.trim()},
        options: Options(headers: _authHeader),
      );
      final data = res.data['data'] as Map<String, dynamic>;
      return ChatMessage.pending(
        id: data['id'] as int,
        chatId: data['chat_id'] as int,
        question: data['question'] as String,
        createdAt: data['created_at'] as String,
      );
    } catch (_) {
      return null;
    }
  }

  Future<ChatMessage?> regenerateResponse(int messageId) async {
    try {
      final res = await _dio.post(
        '/chat/regenerate/$messageId',
        options: Options(headers: _authHeader),
      );
      final data = res.data['data'] as Map<String, dynamic>;
      return ChatMessage.pending(
        id: data['id'] as int,
        chatId: data['chat_id'] as int,
        question: data['question'] as String,
        createdAt: data['created_at'] as String,
      );
    } catch (_) {
      return null;
    }
  }

  Future<ChatMessage?> fetchMessage(int detailId) async {
    try {
      final res = await _dio.get(
        '/chat/message/$detailId',
        options: Options(headers: _authHeader),
      );
      final jsonData = res.data['data'] as Map<String, dynamic>;
      return ChatMessage.fromJson(jsonData);
    } catch (e) {
      print('❌ Error fetching message: $e');
      return null;
    }
  }

  Future<bool> stopGeneration(int detailId) async {
    try {
      await _dio.post(
        '/chat/stop/$detailId',
        options: Options(headers: _authHeader),
      );
      return true;
    } catch (_) {
      return false;
    }
  }

  Future<bool> deleteTopic(int topicId) async {
    try {
      await _dio.delete('/topics/$topicId', options: Options(headers: _authHeader));
      return true;
    } catch (_) {
      return false;
    }
  }

  Future<bool> renameTopic(int topicId, String newTitle) async {
    final trimmed = newTitle.trim();
    if (trimmed.isEmpty) return false;
    try {
      await _dio.patch(
        '/topics/$topicId',
        data: {'title': trimmed},
        options: Options(headers: _authHeader),
      );
      return true;
    } catch (_) {
      return false;
    }
  }

  // -------------------------------------------------------------------------
  // Text-to-Speech (TTS) Methods
  // -------------------------------------------------------------------------

  Future<void> playTTS(int detailId) async {
    try {
      await _audioPlayer.stop();
      playingTtsId.value = detailId;
      _currentTtsRequestDetailId = detailId;

      final response = await _dio.get(
        '/chat/message/$detailId/tts',
        options: Options(
          headers: _authHeader,
          responseType: ResponseType.bytes,
        ),
      );

      if (_currentTtsRequestDetailId != detailId) return;

      final List<int> audioData = response.data;
      final Uint8List uint8Bytes = Uint8List.fromList(audioData);

      await _audioPlayer.play(BytesSource(uint8Bytes));
    } catch (e) {
      if (_currentTtsRequestDetailId == detailId) {
        playingTtsId.value = null;
      }
      print('❌ Error playing TTS: $e');
      throw Exception('Gagal memutar audio TTS.');
    }
  }

  Future<void> stopTTS() async {
    try {
      _currentTtsRequestDetailId = null;
      await _audioPlayer.stop();
      playingTtsId.value = null;
    } catch (e) {
      debugPrint('Error stopping TTS: $e');
    }
  }

  // -------------------------------------------------------------------------
  // Knowledge Upload Methods
  // -------------------------------------------------------------------------

  /// Upload beberapa PDF sekaligus — satu request per file (backend tetap single).
  /// Mengembalikan list hasil per file dalam urutan yang sama.
  /// [onProgress] dipanggil setiap file selesai: (selesai, total).
  Future<List<Map<String, dynamic>?>> uploadPdfs({
    required List<PdfUploadFile> files,
    void Function(int done, int total)? onProgress,
  }) async {
    final results = <Map<String, dynamic>?>[];
    for (var i = 0; i < files.length; i++) {
      final f = files[i];
      try {
        final formData = FormData.fromMap({
          'file': MultipartFile.fromBytes(
            f.bytes,
            filename: f.name,
            contentType: http.MediaType('application', 'pdf'),
          ),
          if (f.judul != null && f.judul!.isNotEmpty) 'judul': f.judul,
          if (f.penulis != null && f.penulis!.isNotEmpty) 'penulis': f.penulis,
          if (f.tahun != null && f.tahun!.isNotEmpty) 'tahun': f.tahun,
          'embedder_type': f.embedderType,
        });

        final res = await _dio.post(
          '/knowledge/upload',
          data: formData,
          options: Options(
            headers: _authHeader,
            sendTimeout: const Duration(minutes: 5),
            receiveTimeout: const Duration(minutes: 5),
          ),
        );
        results.add(res.data as Map<String, dynamic>);
      } on DioException catch (e) {
        if (e.response?.statusCode == 401 || e.response?.statusCode == 403) {
          _forceLogout();
        }
        results.add(e.response?.data as Map<String, dynamic>?);
      } catch (_) {
        results.add(null);
      }
      onProgress?.call(i + 1, files.length);
    }
    return results;
  }

  // -------------------------------------------------------------------------
  // Model Selector Methods
  // -------------------------------------------------------------------------

  /// Ambil daftar model Groq yang tersedia dari backend
  Future<List<Map<String, dynamic>>> getAvailableModels() async {
    try {
      final resp = await _dio.get('/models', options: Options(headers: _authHeader));
      final data = resp.data;
      if (data['success'] == true) {
        return List<Map<String, dynamic>>.from(
          (data['models'] as List).map((m) => Map<String, dynamic>.from(m)),
        );
      }
      return [];
    } catch (e) {
      print('❌ Error getting available models: $e');
      return [];
    }
  }

  /// Get active model dari backend (untuk fallback)
  Future<Map<String, dynamic>?> getActiveModelFromBackend() async {
    try {
      final res = await _dio.get(
        '/models/active',
        options: Options(headers: _authHeader),
      );
      if (res.data['success'] == true) {
        return {
          'model_id': res.data['data']['model_id'],
          'mode': res.data['data']['mode'],
        };
      }
      return null;
    } catch (e) {
      print('❌ Error getting active model: $e');
      return null;
    }
  }

  /// Alias untuk kompatibilitas dengan kode lama yang memanggil getLocalModels
  Future<List<Map<String, dynamic>>> getLocalModels() async {
    return getAvailableModels();
  }

  // -------------------------------------------------------------------------
  // SSE Methods
  // -------------------------------------------------------------------------

  Stream<SseEvent> subscribeToStream(int detailId) {
    final url = '$_kBaseUrl/chat/stream/$detailId';
    final token = _accessToken ?? '';
    return SseClient.subscribe(url, token);
  }

  void dispose() {
    _tokenTimer?.cancel();
    _audioPlayer.dispose();
  }
}

// ---------------------------------------------------------------------------
// SSE Tracker (Helper untuk UI)
// ---------------------------------------------------------------------------

class SseTracker {
  final int detailId;
  StreamSubscription<SseEvent>? sseSub;

  SseTracker({required this.detailId});

  void cancel() {
    sseSub?.cancel();
    sseSub = null;
  }
}

// ---------------------------------------------------------------------------
// Model untuk Multi-file Upload
// ---------------------------------------------------------------------------

class PdfUploadFile {
  final Uint8List bytes;
  final String name;
  final String? judul;
  final String? penulis;
  final String? tahun;
  /// 'improved' (default) atau 'raw'
  final String embedderType;

  const PdfUploadFile({
    required this.bytes,
    required this.name,
    this.judul,
    this.penulis,
    this.tahun,
    this.embedderType = 'improved',
  });
}