import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Storage abstraction:
/// - Web/Chrome → SharedPreferences (localStorage) — persisten lintas restart browser
/// - Android    → FlutterSecureStorage (EncryptedSharedPreferences) — enkripsi hardware
class TokenStorage {
  static final FlutterSecureStorage _secureStorage = FlutterSecureStorage(
    aOptions: const AndroidOptions(
      encryptedSharedPreferences: true,
    ),
    webOptions: const WebOptions(
      dbName: 'agribot_secure',
      publicKey: 'agribot_key',
    ),
  );

  static Future<void> write({required String key, required String value}) async {
    if (kIsWeb) {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(key, value);
      print('✅ TokenStorage: Written $key to SharedPreferences (Web)');
    } else {
      await _secureStorage.write(key: key, value: value);
      print('✅ TokenStorage: Written $key to SecureStorage (Android)');
    }
  }

  static Future<String?> read({required String key}) async {
    String? value;
    if (kIsWeb) {
      final prefs = await SharedPreferences.getInstance();
      value = prefs.getString(key);
      print('📖 TokenStorage: Read $key from SharedPreferences (Web): ${value != null ? 'found' : 'not found'}');
    } else {
      value = await _secureStorage.read(key: key);
      print('📖 TokenStorage: Read $key from SecureStorage (Android): ${value != null ? 'found' : 'not found'}');
    }
    return value;
  }

  static Future<void> delete({required String key}) async {
    if (kIsWeb) {
      final prefs = await SharedPreferences.getInstance();
      await prefs.remove(key);
      print('🗑️ TokenStorage: Deleted $key from SharedPreferences (Web)');
    } else {
      await _secureStorage.delete(key: key);
      print('🗑️ TokenStorage: Deleted $key from SecureStorage (Android)');
    }
  }

  static Future<void> deleteAll() async {
    if (kIsWeb) {
      final prefs = await SharedPreferences.getInstance();
      await Future.wait([
        prefs.remove('access_token'),
        prefs.remove('refresh_token'),
        prefs.remove('user_id'),
        prefs.remove('session_created_at'),
      ]);
      print('🗑️ TokenStorage: Deleted all keys from SharedPreferences (Web)');
    } else {
      await _secureStorage.deleteAll();
      print('🗑️ TokenStorage: Deleted all keys from SecureStorage (Android)');
    }
  }
  
  /// Cek apakah user sudah login
  static Future<bool> isLoggedIn() async {
    final accessToken = await read(key: 'access_token');
    final userId = await read(key: 'user_id');
    return accessToken != null && accessToken.isNotEmpty && userId != null;
  }
  
  /// Dapatkan user ID
  static Future<int?> getUserId() async {
    final userIdStr = await read(key: 'user_id');
    if (userIdStr != null && userIdStr.isNotEmpty) {
      return int.tryParse(userIdStr);
    }
    return null;
  }
}

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