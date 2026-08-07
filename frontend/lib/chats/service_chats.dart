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
// Model Preference Storage
// ---------------------------------------------------------------------------

class ModelPreferenceStorage {
  static const String _keySelectedModel = 'selected_model_id';
  static const String _keySelectedMode = 'selected_llm_mode'; 
  
  static Future<void> saveSelectedModel(String modelId) async {
    await TokenStorage.write(key: _keySelectedModel, value: modelId);
  }
  
  static Future<String?> getSelectedModel() async {
    return await TokenStorage.read(key: _keySelectedModel);
  }
  
  static Future<void> saveSelectedMode(String mode) async {
    await TokenStorage.write(key: _keySelectedMode, value: mode);
  }
  
  static Future<String?> getSelectedMode() async {
    return await TokenStorage.read(key: _keySelectedMode);
  }
  
  static Future<void> clearModelPreference() async {
    await TokenStorage.delete(key: _keySelectedModel);
    await TokenStorage.delete(key: _keySelectedMode);
  }
}

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
  Uint8List? localImageBytes; // <--- DITAMBAHKAN UNTUK VISION RAG

  ChatMessage({
    required this.id,
    required this.chatId,
    required this.question,
    required this.response,
    required this.processingStatus,
    required this.createdAt,
    this.localImageBytes,
  });

  ChatMessage copyWith({
    String? response,
    String? processingStatus,
    Uint8List? localImageBytes,
  }) {
    return ChatMessage(
      id: id,
      chatId: chatId,
      question: question,
      response: response ?? this.response,
      processingStatus: processingStatus ?? this.processingStatus,
      createdAt: createdAt,
      localImageBytes: localImageBytes ?? this.localImageBytes,
    );
  }

  factory ChatMessage.pending({
    required int id,
    required int chatId,
    required String question,
    required String createdAt,
    Uint8List? localImageBytes,
  }) =>
      ChatMessage(
        id: id,
        chatId: chatId,
        question: question,
        response: '',
        processingStatus: 'pending',
        createdAt: createdAt,
        localImageBytes: localImageBytes,
      );

  factory ChatMessage.fromJson(Map<String, dynamic> j) => ChatMessage(
        id: j['id'] as int,
        chatId: j['chat_id'] as int,
        question: j['question'] as String,
        response: j['response'] as String? ?? '',
        processingStatus: j['processing_status'] as String? ?? 'pending',
        createdAt: j['created_at'] as String,
        localImageBytes: null, 
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
    } catch (e) {
      throw Exception('SSE connect gagal: $e');
    }

    if (response.statusCode != 200) {
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
          if (line.isEmpty) continue;

          if (line.startsWith('event:')) {
            currentEventType = line.substring(6).trim();
          } else if (line.startsWith('data:')) {
            final data = line.substring(5).trim();
            if (data.isNotEmpty) {
              yield SseEvent(type: currentEventType, data: data);
              currentEventType = 'message';
            }
          }
        }
      }
    } finally {
      client.close();
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
  
  String? _currentModelId;
  String? _currentMode; 

  final AudioPlayer _audioPlayer = AudioPlayer();
  final ValueNotifier<int?> playingTtsId = ValueNotifier<int?>(null);
  int? _currentTtsRequestDetailId;

  String? get accessToken => _accessToken;
  int? get userId => _userId;
  String? get currentModelId => _currentModelId;

  Map<String, dynamic> get _authHeader => {'Authorization': 'Bearer $_accessToken'};

  final VoidCallback? onForceLogout;
  final void Function(String? token)? onTokenUpdated;
  final void Function(String? modelId)? onModelChanged;

  ChatService({
    this.onForceLogout,
    this.onTokenUpdated,
    this.onModelChanged,
  }) {
    _audioPlayer.onPlayerComplete.listen((_) {
      playingTtsId.value = null;
    });
    _audioPlayer.onPlayerStateChanged.listen((state) {
      if (state == PlayerState.stopped || state == PlayerState.completed) {
        playingTtsId.value = null;
      }
    });
  }

  Future<void> initModelPreference() async {
    try {
      _currentModelId = await ModelPreferenceStorage.getSelectedModel();
      _currentMode = await ModelPreferenceStorage.getSelectedMode();
      
      if (_currentModelId != null && _currentMode != null) {
        final success = await _setModelInternal(_currentMode!, path: _currentModelId);
        if (success) {
          onModelChanged?.call(_currentModelId);
        }
      } else {
        await _syncModelFromBackend();
      }
    } catch (e) {
      debugPrint('Error loading model preference: $e');
    }
  }

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
        
        await ModelPreferenceStorage.saveSelectedModel(modelId);
        await ModelPreferenceStorage.saveSelectedMode(mode);
        onModelChanged?.call(modelId);
      } else {
        // ❌ HAPUS hardcode - ambil dari backend atau tetap null
        // Jangan set default di sini, biarkan null
        _currentModelId = null;
        _currentMode = null;
        debugPrint('⚠️ Failed to sync model from backend');
      }
    } catch (e) {
      // ❌ HAPUS hardcode
      _currentModelId = null;
      _currentMode = null;
      debugPrint('⚠️ Error syncing model: $e');
    }
  }

  Future<String?> refreshCurrentModel() async {
    final active = await getActiveModelFromBackend();
    if (active != null) {
      _currentModelId = active['model_id'] as String;
      _currentMode = active['mode'] as String;
      await ModelPreferenceStorage.saveSelectedModel(_currentModelId!);
      await ModelPreferenceStorage.saveSelectedMode(_currentMode!);
      onModelChanged?.call(_currentModelId);
      return _currentModelId;
    }
    return null;
  }

  Future<bool> _setModelInternal(String mode, {String? path}) async {
    try {
      final resp = await _dio.post(
        '/models/set-model',
        data: {'model_id': path},
        options: Options(headers: _authHeader)
      );
      return resp.data['success'] == true;
    } catch (e) {
      return false;
    }
  }

  Future<bool> setModel(String mode, {String? path}) async {
    try {
      final resp = await _dio.post(
        '/models/set-model',
        data: {'model_id': path},
        options: Options(headers: _authHeader)
      );
      
      if (resp.data['success'] == true) {
        _currentModelId = path;
        _currentMode = mode;
        
        if (path != null) {
          await ModelPreferenceStorage.saveSelectedModel(path);
        }
        await ModelPreferenceStorage.saveSelectedMode(mode);
        
        onModelChanged?.call(path);
        return true;
      }
      return false;
    } catch (e) {
      return false;
    }
  }

  Future<String?> getCurrentModel() async {
    if (_currentModelId != null) return _currentModelId;
    await _syncModelFromBackend();
    return _currentModelId;
  }

  Future<String?> getCurrentMode() async {
    if (_currentMode != null) return _currentMode;
    await _syncModelFromBackend();
    return _currentMode;
  }

  Future<String> getRagMode() async {
    try {
      final res = await _dio.get(
        '/rag/mode',
        options: Options(headers: _authHeader),
      );
      if (res.data['success'] == true) {
        return res.data['data']['mode'] as String;
      }
      return 'improved';
    } catch (e) {
      return 'improved';
    }
  }

  Future<bool> setRagMode(String mode) async {
    try {
      final res = await _dio.post(
        '/rag/set-mode',
        data: {'mode': mode},
        options: Options(headers: _authHeader),
      );
      if (res.data['success'] == true) {
        await RAGPreferenceStorage.saveRagMode(mode);
        return true;
      }
      return false;
    } catch (e) {
      return false;
    }
  }

  Future<void> initRagMode() async {
    try {
      String? savedMode = await RAGPreferenceStorage.getSavedRagMode();
      if (savedMode != null) {
        await setRagMode(savedMode);
      } else {
        final currentMode = await getRagMode();
        await RAGPreferenceStorage.saveRagMode(currentMode);
      }
    } catch (e) {}
  }

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
    await initModelPreference();
  }

  void _startTokenTimer() {
    _tokenTimer?.cancel();
    _tokenTimer = Timer.periodic(_kTokenRefreshInterval, (_) => _silentRefresh());
  }

  Future<void> _silentRefresh() async {
    final rt = await TokenStorage.read(key: 'refresh_token');
    if (rt == null || rt.isEmpty) {
      _forceLogout();
      return;
    }
    
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
      } else {
        if (res.statusCode == 401 || res.statusCode == 403) {
          _forceLogout();
        }
      }
    } on DioException catch (e) {
      if (e.response?.statusCode == 401 || e.response?.statusCode == 403) {
        _forceLogout();
      }
    } catch (_) {}
  }

  void _forceLogout() {
    _tokenTimer?.cancel();
    _accessToken = null;
    _userId = null;
    _audioPlayer.stop();
    playingTtsId.value = null;
    onForceLogout?.call();
  }

  Future<void> forceLogout() async {
    _tokenTimer?.cancel();
    await TokenStorage.deleteAll();
    await ModelPreferenceStorage.clearModelPreference(); 
    _accessToken = null;
    _userId = null;
    _currentModelId = null;
    _currentMode = null;
    await _audioPlayer.stop();
    playingTtsId.value = null;
  }

  Future<void> logout() async {
    try {
      await _dio.post('/users/logout', options: Options(headers: _authHeader));
    } catch (_) {}
    await forceLogout();
  }

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

// ── DIPERBARUI UNTUK VISION RAG (SATU ENDPOINT) ─────────────────────────
  Future<ChatMessage?> sendMessage({
    required int? chatId,
    required String question,
    Uint8List? imageBytes,
    String? imageName,
  }) async {
    try {
      Response res;
      if (imageBytes != null) {
        // Mode Vision (Kirim Multipart Data)
        final formData = FormData.fromMap({
          'chat_id': ?chatId,
          'question': question.isEmpty ? 'Tolong jelaskan gambar tanaman ini.' : question,
          'file': MultipartFile.fromBytes(
            imageBytes,
            filename: imageName ?? 'image.jpg',
          ),
        });

        res = await _dio.post(
          '/chat/send', // <--- PERBAIKAN: Ubah dari /chat/vision menjadi /chat/send
          data: formData,
          options: Options(headers: _authHeader),
        );
      } else {
        // Mode Text Biasa (Kirim JSON Raw)
        res = await _dio.post(
          '/chat/send',
          data: {'chat_id': chatId, 'question': question},
          options: Options(headers: _authHeader),
        );
      }

      final data = res.data['data'] as Map<String, dynamic>;
      return ChatMessage.pending(
        id: data['id'] as int,
        chatId: data['chat_id'] as int,
        question: data['question'] as String,
        createdAt: data['created_at'] as String,
        localImageBytes: imageBytes, // Simpan gambar secara lokal untuk pratinjau di bubble
      );
    } catch (e) {
      debugPrint('Error Send Message: $e');
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
      throw Exception('Gagal memutar audio TTS.');
    }
  }

  Future<void> stopTTS() async {
    try {
      _currentTtsRequestDetailId = null;
      await _audioPlayer.stop();
      playingTtsId.value = null;
    } catch (e) {}
  }

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
      return [];
    }
  }

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
      return null;
    }
  }

  Future<List<Map<String, dynamic>>> getLocalModels() async {
    return getAvailableModels();
  }

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

class SseTracker {
  final int detailId;
  StreamSubscription<SseEvent>? sseSub;

  SseTracker({required this.detailId});

  void cancel() {
    sseSub?.cancel();
    sseSub = null;
  }
}

class PdfUploadFile {
  final Uint8List bytes;
  final String name;
  final String? judul;
  final String? penulis;
  final String? tahun;
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