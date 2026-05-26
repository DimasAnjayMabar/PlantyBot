// lib/chats/chats_page.dart
import 'dart:async';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:frontend/chats/service_chats.dart';
import 'package:frontend/chats/sidebar.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:markdown/markdown.dart' as md;
import 'package:url_launcher/url_launcher.dart';
import 'package:file_picker/file_picker.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt; // <-- Import STT

// ---------------------------------------------------------------------------
// Greetings
// ---------------------------------------------------------------------------

const _kGreetings = [
  'Halo! Ada yang bisa saya bantu hari ini? 🌱',
  'Selamat datang! Silakan tanyakan seputar pertanian kepada saya.',
  'Hai! Saya siap membantu menjawab pertanyaan agrikultur Anda.',
  'Halo, petani hebat! Ada pertanyaan seputar tanaman atau lahan?',
  'Selamat datang kembali! Apa yang ingin Anda ketahui hari ini?',
  'Hai! Saya AgriBot — tanyakan apa saja soal pertanian. 🌾',
  'Halo! Butuh saran soal pupuk, hama, atau panen? Saya siap bantu!',
];

// ---------------------------------------------------------------------------
// Platform Helpers
// ---------------------------------------------------------------------------

bool _isMobileDevice(BuildContext context) {
  final width = MediaQuery.of(context).size.width;
  // Web mobile atau aplikasi mobile (lebar < 768)
  return width < 768 ||
      (!kIsWeb &&
          (defaultTargetPlatform == TargetPlatform.android ||
              defaultTargetPlatform == TargetPlatform.iOS));
}

bool _isDesktopDevice(BuildContext context) {
  return !_isMobileDevice(context);
}

// ---------------------------------------------------------------------------
// ChatsPage
// ---------------------------------------------------------------------------

class ChatsPage extends StatefulWidget {
  const ChatsPage({super.key});

  @override
  State<ChatsPage> createState() => _ChatsPageState();
}

class _ChatsPageState extends State<ChatsPage>
    with SingleTickerProviderStateMixin {
  late final ChatService _chatService;

  bool _sidebarOpen = false; // akan di-set ulang di didChangeDependencies
  bool _sidebarInitialized = false;
  late final AnimationController _sidebarCtrl;
  late final Animation<double> _sidebarAnim;

  List<ChatTopic> _topics = [];
  ChatUserProfile? _profile;
  bool _loadingTopics = true;

  int? _activeChatId;
  List<ChatMessage> _messages = [];
  bool _loadingMessages = false;

  bool _sending = false;
  String? _pendingQuestion;

  final Map<int, SseTracker> _trackers = {};

  int? _renamingId;
  String? _renamingTemp;
  String? _greeting;

  final _inputCtrl = TextEditingController();
  final _scrollCtrl = ScrollController();
  final _inputFocus = FocusNode();

  // ── Question Navigator ────────────────────────────────────────────────────
  bool _questionNavOpen = false;
  final Map<int, GlobalKey> _messageKeys = {};

  @override
  void initState() {
    super.initState();

    _sidebarCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 250),
      value: 1.0,
    );
    _sidebarAnim = CurvedAnimation(
      parent: _sidebarCtrl,
      curve: Curves.easeInOut,
    );

    _chatService = ChatService(
      onForceLogout: _handleForceLogout,
      onTokenUpdated: (_) => setState(() {}),
    );

    _initAuth();
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // Set sidebar open state hanya sekali saat pertama kali build
    if (!_sidebarInitialized) {
      _sidebarInitialized = true;
      final isMobile = MediaQuery.of(context).size.width < 768;
      _sidebarOpen = !isMobile; // mobile: tutup, desktop: buka
      if (_sidebarOpen) {
        _sidebarCtrl.value = 1.0;
      } else {
        _sidebarCtrl.value = 0.0;
      }
    }
  }

  @override
  void dispose() {
    _cancelAllTrackers();
    _chatService.dispose();
    _sidebarCtrl.dispose();
    _inputCtrl.dispose();
    _scrollCtrl.dispose();
    _inputFocus.dispose();
    super.dispose();
  }

  // ── Auth ──────────────────────────────────────────────────────────────────

  Future<void> _initAuth() async {
    final isAuthenticated = await _chatService.initAuth();
    if (!isAuthenticated) {
      if (mounted) context.go('/users/login');
      return;
    }
    await Future.wait([_fetchTopics(), _fetchProfile()]);
    _pickGreeting();
  }

  void _handleForceLogout() {
    if (mounted) context.go('/users/login');
  }

  Future<void> _logout() async {
    _cancelAllTrackers();
    await _chatService.logout();
    if (mounted) context.go('/users/login');
  }

  // ── SSE Tracker ───────────────────────────────────────────────────────────

  void _startTracking(int detailId) {
    _trackers[detailId]?.cancel();
    final tracker = SseTracker(detailId: detailId);
    _trackers[detailId] = tracker;

    tracker.sseSub = _chatService
        .subscribeToStream(detailId)
        .listen(
          (event) async {
            if (!mounted) return;

            if (event.type == 'done' ||
                event.type == 'error' ||
                event.type == 'stopped') {
              await _fetchAndApplyMessage(detailId);
              _stopTracking(detailId);
            } else if (event.type == 'timeout') {
              _markDisconnected(detailId);
              _stopTracking(detailId);
            }
          },
          onError: (error) {
            if (mounted) _markDisconnected(detailId);
            _stopTracking(detailId);
          },
          onDone: () {
            if (mounted) {
              final msg = _messages.firstWhere(
                (m) => m.id == detailId,
                orElse: () => ChatMessage(
                  id: detailId,
                  chatId: 0,
                  question: '',
                  response: '',
                  processingStatus: 'pending',
                  createdAt: '',
                ),
              );
              if (msg.isPending) _markDisconnected(detailId);
            }
            _stopTracking(detailId);
          },
          cancelOnError: true,
        );

    Future.delayed(const Duration(seconds: 30), () {
      if (mounted && _trackers.containsKey(detailId)) {
        final msg = _messages.firstWhere(
          (m) => m.id == detailId,
          orElse: () => ChatMessage(
            id: detailId,
            chatId: 0,
            question: '',
            response: '',
            processingStatus: 'pending',
            createdAt: '',
          ),
        );
        if (msg.isPending) _fetchAndApplyMessage(detailId);
      }
    });
  }

  Future<void> _fetchAndApplyMessage(int detailId) async {
    if (!mounted) return;
    final updated = await _chatService.fetchMessage(detailId);
    if (updated != null) {
      _applyMessageUpdate(updated);
    } else {
      _markDisconnected(detailId);
    }
  }

  void _applyMessageUpdate(ChatMessage updated) {
    if (!mounted) return;
    final idx = _messages.indexWhere((m) => m.id == updated.id);
    if (idx != -1) {
      setState(() => _messages[idx] = updated);
      _scrollToBottom();
    }
  }

  void _markDisconnected(int detailId) {
    if (!mounted) return;
    final idx = _messages.indexWhere((m) => m.id == detailId);
    if (idx != -1) {
      setState(() => _messages[idx].processingStatus = 'disconnected');
    }
  }

  void _stopTracking(int detailId) {
    _trackers[detailId]?.cancel();
    _trackers.remove(detailId);
    if (mounted) setState(() {});
  }

  void _cancelAllTrackers() {
    for (final t in _trackers.values) t.cancel();
    _trackers.clear();
  }

  // ── Fetch ─────────────────────────────────────────────────────────────────

  Future<void> _fetchTopics() async {
    final topics = await _chatService.fetchTopics();
    if (mounted) {
      setState(() {
        _topics = topics;
        _loadingTopics = false;
      });
    }
  }

  Future<void> _fetchProfile() async {
    final profile = await _chatService.fetchProfile();
    if (mounted && profile != null) setState(() => _profile = profile);
  }

  Future<void> _fetchMessages(int chatId) async {
    setState(() {
      _loadingMessages = true;
      _messages = [];
    });

    final msgs = await _chatService.fetchMessages(chatId);

    if (!mounted) return;

    if (msgs == null) {
      setState(() => _loadingMessages = false);
      _showSnack('Gagal memuat pesan.');
      return;
    }

    setState(() {
      _messages = msgs;
      _loadingMessages = false;
    });
    _syncMessageKeys();
    _scrollToBottom();

    for (final msg in msgs.where((m) => m.isPending)) {
      _startTracking(msg.id);
    }
  }

  // ── Actions ───────────────────────────────────────────────────────────────

  void _pickGreeting() {
    setState(
      () => _greeting = _kGreetings[Random().nextInt(_kGreetings.length)],
    );
  }

  void _newChat() {
    _cancelAllTrackers();
    _pickGreeting();
    setState(() {
      _activeChatId = null;
      _messages = [];
      _pendingQuestion = null;
      _questionNavOpen = false;
    });
  }

  void _selectTopic(ChatTopic topic) {
    _cancelAllTrackers();
    setState(() {
      _activeChatId = topic.id;
      _greeting = null;
      _pendingQuestion = null;
      _questionNavOpen = false;
    });
    _fetchMessages(topic.id);
    if (MediaQuery.of(context).size.width < 768) _toggleSidebar();
  }

  Future<void> _sendMessage({
    String? overrideText,
    int? replaceDetailId,
  }) async {
    final text = overrideText ?? _inputCtrl.text.trim();
    if (text.isEmpty || _sending) return;

    if (overrideText == null) _inputCtrl.clear();
    setState(() {
      _sending = true;
      _pendingQuestion = replaceDetailId == null ? text : null;
    });
    _scrollToBottom();

    final msg = await _chatService.sendMessage(
      chatId: _activeChatId,
      question: text,
    );

    if (msg == null) {
      setState(() {
        _pendingQuestion = null;
        _sending = false;
      });
      _showSnack('Gagal mengirim pesan. Coba lagi.');
      return;
    }

    if (_activeChatId == null) {
      setState(() {
        _activeChatId = msg.chatId;
        _greeting = null;
        _pendingQuestion = null;
        _messages.add(msg);
        _sending = false;
      });
      _syncMessageKeys();
      await _fetchTopics();
    } else if (replaceDetailId != null) {
      final idx = _messages.indexWhere((m) => m.id == replaceDetailId);
      setState(() {
        _pendingQuestion = null;
        if (idx != -1) {
          _messages[idx] = msg;
        } else {
          _messages.add(msg);
        }
        _sending = false;
      });
      _syncMessageKeys();
    } else {
      setState(() {
        _pendingQuestion = null;
        _messages.add(msg);
        _sending = false;
      });
      _syncMessageKeys();
    }

    _scrollToBottom();
    _startTracking(msg.id);
  }

  Future<void> _uploadPdfs(
    List<PdfUploadFile> files, {
    void Function(int done, int total)? onProgress,
  }) async {
    final results = await _chatService.uploadPdfs(
      files: files,
      onProgress: onProgress,
    );
    if (!mounted) return;

    final successCount = results.where((r) => r?['success'] == true).length;
    final failCount    = results.length - successCount;

    String message;
    Color  color;
    if (failCount == 0) {
      message = successCount == 1
          ? 'PDF "${files.first.name}" diterima dan sedang diproses.'
          : '$successCount PDF berhasil diterima dan sedang diproses.';
      color = const Color(0xFF16DB65);
    } else if (successCount == 0) {
      final firstFail = results.firstWhere((r) => r?['success'] != true);
      message = firstFail?['detail'] as String? ??
                firstFail?['message'] as String? ??
                'Gagal mengunggah semua file.';
      color = const Color(0xFFFF4444);
    } else {
      message = '$successCount berhasil, $failCount gagal. Periksa ulang file yang gagal.';
      color = const Color(0xFFFFAA00);
    }

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message, style: GoogleFonts.poppins(fontSize: 13)),
        backgroundColor: color,
        behavior: SnackBarBehavior.floating,
        duration: const Duration(seconds: 5),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      ),
    );
  }

  Future<void> _editMessage(ChatMessage msg, String newQuestion) async {
    if (_sending) return;
    setState(() => _sending = true);

    final updated = await _chatService.editMessage(msg.id, newQuestion);

    if (updated != null) {
      final idx = _messages.indexWhere((m) => m.id == msg.id);
      if (idx != -1) setState(() => _messages[idx] = updated);
      _startTracking(msg.id);
    } else {
      _showSnack('Gagal mengedit pesan.');
    }

    if (mounted) setState(() => _sending = false);
  }

  Future<void> _regenerateResponse(ChatMessage msg) async {
    if (_sending) return;
    setState(() => _sending = true);

    final updated = await _chatService.regenerateResponse(msg.id);

    if (updated != null) {
      final idx = _messages.indexWhere((m) => m.id == msg.id);
      if (idx != -1) setState(() => _messages[idx] = updated);
      _startTracking(msg.id);
    } else {
      _showSnack('Gagal regenerate jawaban.');
    }

    if (mounted) setState(() => _sending = false);
  }

  Future<void> _stopGeneration(int detailId) async {
    final success = await _chatService.stopGeneration(detailId);
    if (!success && mounted) {
      _showSnack('Gagal menghentikan generate.');
    }
  }

  Future<void> _copyText(String text) async {
    await Clipboard.setData(ClipboardData(text: text));
    _showSnack('Disalin ke clipboard.');
  }

  Future<void> _playTTS(ChatMessage msg) async {
    if (msg.response.isEmpty) {
      _showSnack('Belum ada jawaban untuk dibacakan.');
      return;
    }
    _showSnack('Memuat suara...');

    try {
      await _chatService.playTTS(msg.id);
    } catch (e) {
      _showSnack('Gagal memutar suara.');
    }
  }

  Future<void> _stopTTS() async {
    try {
      await _chatService.stopTTS();
    } catch (e) {
      _showSnack('Gagal menghentikan suara.');
    }
  }

  Future<void> _resendMessage(ChatMessage msg) async {
    await _sendMessage(overrideText: msg.question, replaceDetailId: msg.id);
  }

  Future<void> _deleteTopic(ChatTopic topic) async {
    final success = await _chatService.deleteTopic(topic.id);
    if (success) {
      _cancelAllTrackers();
      setState(() {
        _topics.removeWhere((t) => t.id == topic.id);
        if (_activeChatId == topic.id) {
          _activeChatId = null;
          _messages = [];
          _pendingQuestion = null;
          _pickGreeting();
        }
      });
    } else {
      _showSnack('Gagal menghapus topik.');
    }
  }

  Future<void> _renameTopic(ChatTopic topic, String newTitle) async {
    final trimmed = newTitle.trim();
    if (trimmed.isEmpty) return;
    final success = await _chatService.renameTopic(topic.id, trimmed);
    if (success) {
      setState(() {
        topic.title = trimmed;
        _renamingId = null;
      });
    } else {
      _showSnack('Gagal mengganti judul.');
    }
  }

  void _toggleSidebar() {
    setState(() => _sidebarOpen = !_sidebarOpen);
    _sidebarOpen ? _sidebarCtrl.forward() : _sidebarCtrl.reverse();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollCtrl.hasClients) {
        _scrollCtrl.animateTo(
          _scrollCtrl.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  void _scrollToMessage(int detailId) {
    final key = _messageKeys[detailId];
    if (key?.currentContext != null) {
      Scrollable.ensureVisible(
        key!.currentContext!,
        duration: const Duration(milliseconds: 400),
        curve: Curves.easeOut,
        alignment: 0.05,
      );
    }
    setState(() => _questionNavOpen = false);
  }

  void _syncMessageKeys() {
    final currentIds = _messages.map((m) => m.id).toSet();
    _messageKeys.removeWhere((id, _) => !currentIds.contains(id));
    for (final msg in _messages) {
      _messageKeys.putIfAbsent(msg.id, () => GlobalKey());
    }
  }

  void _showSnack(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(msg, style: GoogleFonts.poppins(fontSize: 13)),
        backgroundColor: const Color(0xFF111111),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      ),
    );
  }

  // ── Build ─────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final int? pendingDetailId = _trackers.isNotEmpty
        ? _trackers.keys.first
        : null;
    final isMobile = MediaQuery.of(context).size.width < 768;

    // Widget konten utama (topbar + body + input)
    final mainContent = Column(
      children: [
        _ChatTopBar(
          sidebarOpen: _sidebarOpen,
          onToggleSidebar: _toggleSidebar,
          title: _activeChatId != null
              ? _topics
                    .firstWhere(
                      (t) => t.id == _activeChatId,
                      orElse: () => ChatTopic(
                        id: 0,
                        title: 'Chat',
                        createdAt: '',
                      ),
                    )
                    .title
              : 'Chat Baru',
          hasPending: _trackers.isNotEmpty,
        ),
        Expanded(child: _buildBody()),
        _InputBar(
          controller: _inputCtrl,
          focusNode: _inputFocus,
          sending: _sending,
          onSend: () => _sendMessage(),
          onUploadPdfs: _uploadPdfs,
          onSetModel: (mode, {path}) => _chatService.setModel(mode, path: path),
          onGetModels: () => _chatService.getLocalModels(),
          pendingDetailId: pendingDetailId,
          onStop: pendingDetailId != null
              ? () => _stopGeneration(pendingDetailId)
              : null,
        ),
      ],
    );

    final sidebarWidget = ChatSidebar(
      topics: _topics,
      loading: _loadingTopics,
      activeChatId: _activeChatId,
      profile: _profile,
      renamingId: _renamingId,
      renamingTemp: _renamingTemp,
      onNewChat: _newChat,
      onSelectTopic: _selectTopic,
      onDeleteTopic: _deleteTopic,
      onStartRename: (t) => setState(() {
        _renamingId = t.id;
        _renamingTemp = t.title;
      }),
      onConfirmRename: (t, v) => _renameTopic(t, v),
      onCancelRename: () => setState(() => _renamingId = null),
      onRenameChange: (v) => setState(() => _renamingTemp = v),
      onProfileTap: () => context.go('/user_profile'),
      onLogout: _logout,
    );

    return Scaffold(
      backgroundColor: const Color(0xFF020202),
      body: isMobile
          // ── Mobile: sidebar sebagai overlay di atas konten ──────────────
          ? Stack(
              children: [
                // Konten utama selalu full width
                mainContent,

                // Overlay gelap saat sidebar terbuka
                if (_sidebarOpen)
                  GestureDetector(
                    onTap: _toggleSidebar,
                    child: Container(
                      color: Colors.black54,
                    ),
                  ),

                // Sidebar geser dari kiri
                AnimatedBuilder(
                  animation: _sidebarAnim,
                  builder: (context, child) {
                    return Transform.translate(
                      offset: Offset(kSidebarWidth * (_sidebarAnim.value - 1), 0),
                      child: child,
                    );
                  },
                  child: SizedBox(
                    width: kSidebarWidth,
                    height: double.infinity,
                    child: sidebarWidget,
                  ),
                ),
              ],
            )
          // ── Desktop: sidebar di samping konten ──────────────────────────
          : Row(
              children: [
                SizeTransition(
                  sizeFactor: _sidebarAnim,
                  axis: Axis.horizontal,
                  child: sidebarWidget,
                ),
                Expanded(child: mainContent),
              ],
            ),
    );
  }

  Widget _buildBody() {
    if (_loadingMessages) {
      return const Center(
        child: CircularProgressIndicator(
          color: Color(0xFF16DB65),
          strokeWidth: 2,
        ),
      );
    }
    if (_activeChatId == null &&
        _messages.isEmpty &&
        _pendingQuestion == null) {
      return _GreetingView(greeting: _greeting ?? _kGreetings[0]);
    }
    if (_messages.isEmpty && _pendingQuestion == null) {
      return const _GreetingView(
        greeting: 'Topik ini masih kosong. Mulai percakapan! 💬',
      );
    }

    final isMobile = _isMobileDevice(context);
    final itemCount = _messages.length + (_pendingQuestion != null ? 1 : 0);

    return Stack(
      children: [
        ListView.builder(
          controller: _scrollCtrl,
          padding: const EdgeInsets.fromLTRB(24, 24, 24, 8),
          itemCount: itemCount,
          itemBuilder: (_, i) {
            if (_pendingQuestion != null && i == _messages.length) {
              return _PendingBubble(question: _pendingQuestion!);
            }

            final msg = _messages[i];
            // Assign GlobalKey ke setiap message item
            final itemKey = _messageKeys.putIfAbsent(msg.id, () => GlobalKey());

            if (msg.isPending) {
              return KeyedSubtree(
                key: itemKey,
                child: _PendingBubble(question: msg.question),
              );
            }

            if (msg.isDisconnected) {
              return KeyedSubtree(
                key: itemKey,
                child: _DisconnectedBubble(
                  message: msg,
                  onResend: () => _resendMessage(msg),
                ),
              );
            }

            if (msg.isFailed) {
              return KeyedSubtree(
                key: itemKey,
                child: _ErrorBubble(text: msg.response),
              );
            }

            if (msg.isStopped) {
              return KeyedSubtree(
                key: itemKey,
                child: ValueListenableBuilder<int?>(
                  valueListenable: _chatService.playingTtsId,
                  builder: (context, playingId, _) {
                    final isPlaying = playingId == msg.id;
                    return _StoppedBubble(
                      response: msg.response,
                      onRegenerate: () => _regenerateResponse(msg),
                      onCopyAnswer: () => _copyText(msg.response),
                      isPlayingTts: isPlaying,
                      onToggleTTS: () {
                        if (isPlaying) {
                          _stopTTS();
                        } else {
                          _playTTS(msg);
                        }
                      },
                    );
                  },
                ),
              );
            }

            // Status 'done' — jawaban lengkap
            return KeyedSubtree(
              key: itemKey,
              child: ValueListenableBuilder<int?>(
                valueListenable: _chatService.playingTtsId,
                builder: (context, playingId, _) {
                  final isPlaying = playingId == msg.id;
                  return _MessagePair(
                    message: msg,
                    onEdit: (newQ) => _editMessage(msg, newQ),
                    onRegenerate: () => _regenerateResponse(msg),
                    onCopyQuestion: () => _copyText(msg.question),
                    onCopyAnswer: () => _copyText(msg.response),
                    isPlayingTts: isPlaying,
                    onToggleTTS: () {
                      if (isPlaying) {
                        _stopTTS();
                      } else {
                        _playTTS(msg);
                      }
                    },
                  );
                },
              ),
            );
          },
        ),

        // ── Question Navigator ──────────────────────────────────────────────
        if (_messages.isNotEmpty)
          _QuestionNavigator(
            messages: _messages,
            isOpen: _questionNavOpen,
            onToggle: () => setState(() => _questionNavOpen = !_questionNavOpen),
            onSelect: _scrollToMessage,
          ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Question Navigator
// ---------------------------------------------------------------------------

class _QuestionNavigator extends StatefulWidget {
  const _QuestionNavigator({
    required this.messages,
    required this.isOpen,
    required this.onToggle,
    required this.onSelect,
  });

  final List<ChatMessage> messages;
  final bool isOpen;
  final VoidCallback onToggle;
  final void Function(int detailId) onSelect;

  @override
  State<_QuestionNavigator> createState() => _QuestionNavigatorState();
}

class _QuestionNavigatorState extends State<_QuestionNavigator>
    with SingleTickerProviderStateMixin {
  late final AnimationController _animCtrl;
  late final Animation<double> _panelAnim;

  // Pagination
  static const int _kPageSize = 5;
  int _page = 0;

  bool get _panelVisible => widget.isOpen;

  List<ChatMessage> get _visibleMessages =>
      widget.messages
          .where((m) => m.question.isNotEmpty)
          .toList();

  int get _totalPages =>
      (_visibleMessages.length / _kPageSize).ceil().clamp(1, 999);

  List<ChatMessage> get _currentPageMessages {
    final all = _visibleMessages;
    final start = _page * _kPageSize;
    final end = (start + _kPageSize).clamp(0, all.length);
    return all.sublist(start, end);
  }

  @override
  void initState() {
    super.initState();
    _animCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 200),
    );
    _panelAnim = CurvedAnimation(parent: _animCtrl, curve: Curves.easeOut);
  }

  @override
  void didUpdateWidget(_QuestionNavigator old) {
    super.didUpdateWidget(old);
    if (widget.messages.length != old.messages.length) {
      final newTotal =
          (_visibleMessages.length / _kPageSize).ceil().clamp(1, 999);
      if (_page >= newTotal) _page = newTotal - 1;
    }
    if (widget.isOpen != old.isOpen) {
      widget.isOpen ? _animCtrl.forward() : _animCtrl.reverse();
    }
  }

  @override
  void dispose() {
    _animCtrl.dispose();
    super.dispose();
  }

  void _prevPage() {
    if (_page > 0) setState(() => _page--);
  }

  void _nextPage() {
    if (_page < _totalPages - 1) setState(() => _page++);
  }

  Widget _buildItem(int i) {
    final msg = _currentPageMessages[i];
    final globalIdx = _page * _kPageSize + i;
    final preview = msg.question.length > 38
        ? '${msg.question.substring(0, 38)}...'
        : msg.question;
    return InkWell(
      onTap: () => widget.onSelect(msg.id),
      borderRadius: BorderRadius.circular(6),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(
              width: 20,
              child: Text(
                '${globalIdx + 1}.',
                style: GoogleFonts.poppins(
                  fontSize: 11,
                  color: const Color(0xFF16DB65),
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
            const SizedBox(width: 4),
            Expanded(
              child: Text(
                preview,
                style: GoogleFonts.poppins(
                  fontSize: 11,
                  color: const Color(0xFFCCCCCC),
                  height: 1.4,
                ),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Positioned(
      left: 0,
      top: 0,
      right: 0,
      bottom: 0,
      child: Stack(
        children: [
          // ── Barrier: tap di luar panel menutup navigator ──────────────
          if (_panelVisible)
            Positioned.fill(
              child: GestureDetector(
                onTap: widget.onToggle,
                behavior: HitTestBehavior.translucent,
              ),
            ),

          // ── Strip + Panel ─────────────────────────────────────────────
          Positioned(
            left: 0,
            top: 0,
            bottom: 0,
            child: Row(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                // Strip Toggle Button
                GestureDetector(
                  onTap: widget.onToggle,
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 200),
                    width: 16,
                    height: 72,
                    decoration: BoxDecoration(
                      color: _panelVisible
                          ? const Color(0xFF16DB65).withOpacity(0.15)
                          : const Color(0xFF1A1A1A).withOpacity(0.85),
                      borderRadius: const BorderRadius.only(
                        topRight: Radius.circular(8),
                        bottomRight: Radius.circular(8),
                      ),
                      border: Border(
                        top: BorderSide(
                          color: _panelVisible
                              ? const Color(0xFF16DB65).withOpacity(0.4)
                              : const Color(0xFF2A2A2A),
                        ),
                        right: BorderSide(
                          color: _panelVisible
                              ? const Color(0xFF16DB65).withOpacity(0.4)
                              : const Color(0xFF2A2A2A),
                        ),
                        bottom: BorderSide(
                          color: _panelVisible
                              ? const Color(0xFF16DB65).withOpacity(0.4)
                              : const Color(0xFF2A2A2A),
                        ),
                      ),
                    ),
                    child: Center(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: List.generate(
                          3,
                          (i) => Container(
                            width: 3,
                            height: 3,
                            margin: const EdgeInsets.symmetric(vertical: 2),
                            decoration: BoxDecoration(
                              color: _panelVisible
                                  ? const Color(0xFF16DB65)
                                  : const Color(0xFF555555),
                              shape: BoxShape.circle,
                            ),
                          ),
                        ),
                      ),
                    ),
                  ),
                ),

                // Panel daftar pertanyaan
                SizeTransition(
                  sizeFactor: _panelAnim,
                  axis: Axis.horizontal,
                  child: FadeTransition(
                    opacity: _panelAnim,
                    child: Container(
                      width: 200,
                      height: 150,
                      margin: const EdgeInsets.only(left: 4),
                      decoration: BoxDecoration(
                        color: const Color(0xFF0F0F0F),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: const Color(0xFF2A2A2A)),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.4),
                            blurRadius: 12,
                            offset: const Offset(2, 4),
                          ),
                        ],
                      ),
                      child: Column(
                        children: [
                          // Header
                          Padding(
                            padding: const EdgeInsets.fromLTRB(12, 10, 8, 6),
                            child: Row(
                              children: [
                                const Icon(
                                  Icons.format_list_bulleted_rounded,
                                  size: 13,
                                  color: Color(0xFF16DB65),
                                ),
                                const SizedBox(width: 6),
                                Text(
                                  'Daftar Pertanyaan',
                                  style: GoogleFonts.poppins(
                                    fontSize: 11,
                                    fontWeight: FontWeight.w600,
                                    color: const Color(0xFF16DB65),
                                  ),
                                ),
                                const Spacer(),
                                if (_totalPages > 1)
                                  Text(
                                    '${_page + 1}/$_totalPages',
                                    style: GoogleFonts.poppins(
                                      fontSize: 10,
                                      color: const Color(0xFF555555),
                                    ),
                                  ),
                              ],
                            ),
                          ),
                          const Divider(height: 1, color: Color(0xFF1A1A1A)),

                          // List pertanyaan — hybrid height
                          if (_visibleMessages.length > 3)
                            Expanded(
                              child: Scrollbar(
                                thumbVisibility: true,
                                child: ListView.builder(
                                  padding: const EdgeInsets.symmetric(vertical: 4),
                                  itemCount: _currentPageMessages.length,
                                  itemBuilder: (_, i) => _buildItem(i),
                                ),
                              ),
                            )
                          else
                            ListView.builder(
                              shrinkWrap: true,
                              padding: const EdgeInsets.symmetric(vertical: 4),
                              itemCount: _currentPageMessages.length,
                              itemBuilder: (_, i) => _buildItem(i),
                            ),

                          // Pagination controls
                          if (_totalPages > 1) ...[
                            const Divider(
                                height: 1, color: Color(0xFF1A1A1A)),
                            Padding(
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 8, vertical: 6),
                              child: Row(
                                mainAxisAlignment:
                                    MainAxisAlignment.spaceBetween,
                                children: [
                                  _NavButton(
                                    icon: Icons.chevron_left_rounded,
                                    onTap: _page > 0 ? _prevPage : null,
                                  ),
                                  Text(
                                    '${_page * _kPageSize + 1}–'
                                    '${_page * _kPageSize + _currentPageMessages.length}'
                                    ' / ${_visibleMessages.length}',
                                    style: GoogleFonts.poppins(
                                      fontSize: 10,
                                      color: const Color(0xFF555555),
                                    ),
                                  ),
                                  _NavButton(
                                    icon: Icons.chevron_right_rounded,
                                    onTap: _page < _totalPages - 1
                                        ? _nextPage
                                        : null,
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ],
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _NavButton extends StatelessWidget {
  const _NavButton({required this.icon, this.onTap});
  final IconData icon;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final enabled = onTap != null;
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 26,
        height: 26,
        decoration: BoxDecoration(
          color: enabled
              ? const Color(0xFF1A1A1A)
              : const Color(0xFF111111),
          borderRadius: BorderRadius.circular(6),
          border: Border.all(
            color: enabled
                ? const Color(0xFF2A2A2A)
                : const Color(0xFF1A1A1A),
          ),
        ),
        child: Icon(
          icon,
          size: 16,
          color: enabled
              ? const Color(0xFFA3A3A3)
              : const Color(0xFF333333),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Chat Top Bar
// ---------------------------------------------------------------------------

class _ChatTopBar extends StatelessWidget {
  const _ChatTopBar({
    required this.sidebarOpen,
    required this.onToggleSidebar,
    required this.title,
    required this.hasPending,
  });

  final bool sidebarOpen;
  final VoidCallback onToggleSidebar;
  final String title;
  final bool hasPending;

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 56,
      padding: const EdgeInsets.symmetric(horizontal: 12),
      decoration: const BoxDecoration(
        color: Color(0xFF0D0D0D),
        border: Border(bottom: BorderSide(color: Color(0xFF1A1A1A))),
      ),
      child: Row(
        children: [
          Tooltip(
            message: sidebarOpen ? 'Sembunyikan Sidebar' : 'Tampilkan Sidebar',
            child: InkWell(
              onTap: onToggleSidebar,
              borderRadius: BorderRadius.circular(8),
              child: Container(
                padding: const EdgeInsets.all(7),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: const Color(0xFF1A1A1A)),
                ),
                child: Icon(
                  sidebarOpen ? Icons.menu_open_rounded : Icons.menu_rounded,
                  size: 18,
                  color: const Color(0xFFA3A3A3),
                ),
              ),
            ),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Text(
              title,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: GoogleFonts.poppins(
                fontSize: 14,
                fontWeight: FontWeight.w600,
                color: Colors.white,
              ),
            ),
          ),
          if (hasPending) ...[
            const SizedBox(width: 8),
            Tooltip(
              message: 'Menunggu respons AI (pipeline aktif)...',
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  _PulsingDot(),
                  const SizedBox(width: 5),
                  Text(
                    'Memproses',
                    style: GoogleFonts.poppins(
                      fontSize: 11,
                      color: const Color(0xFF16DB65),
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class _PulsingDot extends StatefulWidget {
  @override
  State<_PulsingDot> createState() => _PulsingDotState();
}

class _PulsingDotState extends State<_PulsingDot>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => FadeTransition(
    opacity: _ctrl,
    child: Container(
      width: 7,
      height: 7,
      decoration: const BoxDecoration(
        color: Color(0xFF16DB65),
        shape: BoxShape.circle,
      ),
    ),
  );
}

// ---------------------------------------------------------------------------
// Greeting View
// ---------------------------------------------------------------------------

class _GreetingView extends StatelessWidget {
  const _GreetingView({required this.greeting});
  final String greeting;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 40),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 64,
              height: 64,
              decoration: BoxDecoration(
                color: const Color(0x3316DB65),
                shape: BoxShape.circle,
                border: Border.all(
                  color: const Color(0xFF16DB65).withOpacity(0.4),
                  width: 1.5,
                ),
              ),
              child: const Icon(
                Icons.eco_rounded,
                color: Color(0xFF16DB65),
                size: 30,
              ),
            ),
            const SizedBox(height: 20),
            Text(
              greeting,
              textAlign: TextAlign.center,
              style: GoogleFonts.poppins(
                fontSize: 16,
                fontWeight: FontWeight.w500,
                color: Colors.white,
                height: 1.6,
              ),
            ),
            const SizedBox(height: 10),
            Text(
              'Ketik pertanyaan Anda di bawah untuk memulai.',
              textAlign: TextAlign.center,
              style: GoogleFonts.poppins(
                fontSize: 13,
                color: const Color(0xFFA3A3A3),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Message Pair — done / failed / stopped
// ---------------------------------------------------------------------------

class _MessagePair extends StatefulWidget {
  const _MessagePair({
    required this.message,
    required this.onEdit,
    required this.onRegenerate,
    required this.onCopyQuestion,
    required this.onCopyAnswer,
    required this.isPlayingTts,
    required this.onToggleTTS,
  });

  final ChatMessage message;
  final void Function(String) onEdit;
  final VoidCallback onRegenerate;
  final VoidCallback onCopyQuestion;
  final VoidCallback onCopyAnswer;
  final bool isPlayingTts;
  final VoidCallback onToggleTTS;

  @override
  State<_MessagePair> createState() => _MessagePairState();
}

class _MessagePairState extends State<_MessagePair> {
  bool _hovered = false;

  void _showQuestionActions(BuildContext context) {
    final isMobile = _isMobileDevice(context);

    if (isMobile) {
      showModalBottomSheet(
        context: context,
        backgroundColor: const Color(0xFF111111),
        shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
        ),
        builder: (ctx) => SafeArea(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const SizedBox(height: 8),
              Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: const Color(0xFF2A2A2A),
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              const SizedBox(height: 16),
              ListTile(
                leading: const Icon(
                  Icons.edit_outlined,
                  color: Color(0xFF16DB65),
                ),
                title: Text(
                  'Edit pertanyaan',
                  style: GoogleFonts.poppins(color: Colors.white),
                ),
                onTap: () {
                  Navigator.pop(ctx);
                  _showEditDialog(context);
                },
              ),
              ListTile(
                leading: const Icon(
                  Icons.copy_rounded,
                  color: Color(0xFF16DB65),
                ),
                title: Text(
                  'Salin pertanyaan',
                  style: GoogleFonts.poppins(color: Colors.white),
                ),
                onTap: () {
                  Navigator.pop(ctx);
                  widget.onCopyQuestion();
                },
              ),
              const SizedBox(height: 16),
            ],
          ),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final msg = widget.message;
    final isMobile = _isMobileDevice(context);

    return Padding(
      padding: const EdgeInsets.only(bottom: 24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ── User bubble dengan action buttons ───────────────────────────────
          if (isMobile)
            GestureDetector(
              onLongPress: () => _showQuestionActions(context),
              child: _UserBubble(text: msg.question),
            )
          else
            MouseRegion(
              onEnter: (_) => setState(() => _hovered = true),
              onExit: (_) => setState(() => _hovered = false),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  _UserBubble(text: msg.question),
                  AnimatedOpacity(
                    opacity: _hovered && !msg.isStopped ? 1.0 : 0.0,
                    duration: const Duration(milliseconds: 150),
                    child: Padding(
                      padding: const EdgeInsets.only(top: 6, right: 4),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          _ActionChip(
                            icon: Icons.edit_outlined,
                            label: 'Edit',
                            onTap: () => _showEditDialog(context),
                          ),
                          const SizedBox(width: 6),
                          _ActionChip(
                            icon: Icons.copy_rounded,
                            label: 'Salin',
                            onTap: widget.onCopyQuestion,
                          ),
                        ],
                      ),
                    ),
                  ),
                  if (msg.isStopped)
                    AnimatedOpacity(
                      opacity: _hovered ? 1.0 : 0.0,
                      duration: const Duration(milliseconds: 150),
                      child: Padding(
                        padding: const EdgeInsets.only(top: 6, right: 4),
                        child: _ActionChip(
                          icon: Icons.copy_rounded,
                          label: 'Salin',
                          onTap: widget.onCopyQuestion,
                        ),
                      ),
                    ),
                ],
              ),
            ),

          const SizedBox(height: 12),

          // ── AI response area ──────────────────────────────────────────
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _AiBubble(text: msg.response),
              const SizedBox(height: 8),
              _AnswerActions(
                onRegenerate: widget.onRegenerate,
                onCopy: widget.onCopyAnswer,
                isPlayingTts: widget.isPlayingTts,
                onToggleTTS: widget.onToggleTTS,
              ),
            ],
          ),
        ],
      ),
    );
  }

  void _showEditDialog(BuildContext context) {
    final ctrl = TextEditingController(text: widget.message.question);
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: const Color(0xFF111111),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
        title: Text(
          'Edit Pertanyaan',
          style: GoogleFonts.poppins(
            fontSize: 15,
            fontWeight: FontWeight.w600,
            color: Colors.white,
          ),
        ),
        content: SizedBox(
          width: 480,
          child: TextField(
            controller: ctrl,
            autofocus: true,
            maxLines: null,
            style: GoogleFonts.poppins(fontSize: 14, color: Colors.white),
            cursorColor: const Color(0xFF16DB65),
            decoration: InputDecoration(
              filled: true,
              fillColor: const Color(0xFF1A1A1A),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(10),
                borderSide: const BorderSide(color: Color(0xFF2A2A2A)),
              ),
              enabledBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(10),
                borderSide: const BorderSide(color: Color(0xFF2A2A2A)),
              ),
              focusedBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(10),
                borderSide: const BorderSide(
                  color: Color(0xFF16DB65),
                  width: 1.5,
                ),
              ),
            ),
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: Text(
              'Batal',
              style: GoogleFonts.poppins(color: const Color(0xFFA3A3A3)),
            ),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF16DB65),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(8),
              ),
              elevation: 0,
            ),
            onPressed: () {
              final text = ctrl.text.trim();
              if (text.isNotEmpty && text != widget.message.question) {
                Navigator.pop(ctx);
                widget.onEdit(text);
              }
            },
            child: Text(
              'Simpan',
              style: GoogleFonts.poppins(
                color: Colors.black,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Action Chip
// ---------------------------------------------------------------------------

class _ActionChip extends StatelessWidget {
  const _ActionChip({
    required this.icon,
    required this.label,
    required this.onTap,
    this.active = false,
  });

  final IconData icon;
  final String label;
  final VoidCallback onTap;
  final bool active;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(6),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: active 
              ? const Color(0xFF16DB65).withOpacity(0.15) 
              : const Color(0xFF1A1A1A),
          borderRadius: BorderRadius.circular(6),
          border: Border.all(
            color: active 
                ? const Color(0xFF16DB65).withOpacity(0.5) 
                : const Color(0xFF2A2A2A),
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              icon, 
              size: 13, 
              color: active ? const Color(0xFF16DB65) : const Color(0xFFA3A3A3)
            ),
            const SizedBox(width: 5),
            Text(
              label,
              style: GoogleFonts.poppins(
                fontSize: 11,
                color: active ? const Color(0xFF16DB65) : const Color(0xFFA3A3A3),
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Answer Actions 
// ---------------------------------------------------------------------------

class _AnswerActions extends StatelessWidget {
  const _AnswerActions({
    required this.onRegenerate,
    required this.onCopy,
    required this.isPlayingTts,
    required this.onToggleTTS,
  });

  final VoidCallback onRegenerate;
  final VoidCallback onCopy;
  final bool isPlayingTts;
  final VoidCallback onToggleTTS;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(left: 40), 
      child: Wrap(
        spacing: 8,
        runSpacing: 8,
        children: [
          _ActionChip(
            icon: Icons.refresh_rounded,
            label: 'Generate ulang',
            onTap: onRegenerate,
          ),
          _ActionChip(
            icon: Icons.copy_rounded,
            label: 'Salin',
            onTap: onCopy,
          ),
          _ActionChip(
            icon: isPlayingTts ? Icons.volume_off_rounded : Icons.volume_up_rounded,
            label: isPlayingTts ? 'Stop Suara' : 'Dengarkan',
            onTap: onToggleTTS,
            active: isPlayingTts,
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Pending Bubble
// ---------------------------------------------------------------------------

class _PendingBubble extends StatefulWidget {
  const _PendingBubble({required this.question});
  final String question;

  @override
  State<_PendingBubble> createState() => _PendingBubbleState();
}

class _PendingBubbleState extends State<_PendingBubble>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;
  late final Animation<double> _anim;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    )..repeat(reverse: true);
    _anim = CurvedAnimation(parent: _ctrl, curve: Curves.easeInOut);
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _UserBubble(text: widget.question),
          const SizedBox(height: 12),
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _AiAvatar(),
              const SizedBox(width: 10),
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 14,
                ),
                decoration: BoxDecoration(
                  color: const Color(0xFF111111),
                  borderRadius: const BorderRadius.only(
                    topLeft: Radius.circular(4),
                    topRight: Radius.circular(16),
                    bottomLeft: Radius.circular(16),
                    bottomRight: Radius.circular(16),
                  ),
                  border: Border.all(color: const Color(0xFF1A1A1A)),
                ),
                child: FadeTransition(
                  opacity: _anim,
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: List.generate(
                      3,
                      (i) => Padding(
                        padding: EdgeInsets.only(left: i == 0 ? 0 : 5),
                        child: Container(
                          width: 7,
                          height: 7,
                          decoration: const BoxDecoration(
                            color: Color(0xFF16DB65),
                            shape: BoxShape.circle,
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Disconnected Bubble
// ---------------------------------------------------------------------------

class _DisconnectedBubble extends StatelessWidget {
  const _DisconnectedBubble({required this.message, required this.onResend});
  final ChatMessage message;
  final VoidCallback onResend;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _UserBubble(text: message.question),
          const SizedBox(height: 12),
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                width: 30,
                height: 30,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: const Color(0x33FF9800),
                  border: Border.all(
                    color: const Color(0xFFFF9800).withOpacity(0.4),
                  ),
                ),
                child: const Icon(
                  Icons.wifi_off_rounded,
                  color: Color(0xFFFF9800),
                  size: 15,
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 14,
                  ),
                  decoration: BoxDecoration(
                    color: const Color(0xFF1A1200),
                    borderRadius: const BorderRadius.only(
                      topLeft: Radius.circular(4),
                      topRight: Radius.circular(16),
                      bottomLeft: Radius.circular(16),
                      bottomRight: Radius.circular(16),
                    ),
                    border: Border.all(
                      color: const Color(0xFFFF9800).withOpacity(0.3),
                    ),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Koneksi terputus sebelum jawaban diterima.',
                        style: GoogleFonts.poppins(
                          fontSize: 13,
                          color: const Color(0xFFFFB74D),
                          height: 1.5,
                        ),
                      ),
                      const SizedBox(height: 10),
                      GestureDetector(
                        onTap: onResend,
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 14,
                            vertical: 8,
                          ),
                          decoration: BoxDecoration(
                            color: const Color(0xFF2A1A00),
                            borderRadius: BorderRadius.circular(8),
                            border: Border.all(
                              color: const Color(0xFFFF9800).withOpacity(0.5),
                            ),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              const Icon(
                                Icons.refresh_rounded,
                                color: Color(0xFFFF9800),
                                size: 14,
                              ),
                              const SizedBox(width: 6),
                              Text(
                                'Kirim ulang pertanyaan',
                                style: GoogleFonts.poppins(
                                  fontSize: 12,
                                  color: const Color(0xFFFF9800),
                                  fontWeight: FontWeight.w500,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Stopped Bubble
// ---------------------------------------------------------------------------

class _StoppedBubble extends StatelessWidget {
  const _StoppedBubble({
    required this.response,
    required this.onRegenerate,
    required this.onCopyAnswer,
    required this.isPlayingTts,
    required this.onToggleTTS,
  });

  final String response;
  final VoidCallback onRegenerate;
  final VoidCallback onCopyAnswer;
  final bool isPlayingTts;
  final VoidCallback onToggleTTS;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _AiAvatar(),
            const SizedBox(width: 10),
            Expanded(
              child: Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 12,
                ),
                decoration: BoxDecoration(
                  color: const Color(0xFF111111),
                  borderRadius: const BorderRadius.only(
                    topLeft: Radius.circular(4),
                    topRight: Radius.circular(16),
                    bottomLeft: Radius.circular(16),
                    bottomRight: Radius.circular(16),
                  ),
                  border: Border.all(color: const Color(0xFF2A2A2A)),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    if (response.isNotEmpty) ...[
                      MarkdownBody(
                        data: response,
                        selectable: true,
                        extensionSet: md.ExtensionSet.gitHubWeb,
                        onTapLink: (text, href, title) {
                          if (href != null) launchUrl(Uri.parse(href));
                        },
                        styleSheet: _markdownStyleSheet(),
                      ),
                      const Divider(
                        color: Color(0xFF2A2A2A),
                        height: 20,
                        thickness: 1,
                      ),
                    ],
                    Row(
                      children: [
                        const Icon(
                          Icons.stop_circle_outlined,
                          color: Color(0xFFA3A3A3),
                          size: 14,
                        ),
                        const SizedBox(width: 6),
                        Text(
                          'Generate dihentikan',
                          style: GoogleFonts.poppins(
                            fontSize: 12,
                            color: const Color(0xFFA3A3A3),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        _AnswerActions(
          onRegenerate: onRegenerate,
          onCopy: onCopyAnswer,
          isPlayingTts: isPlayingTts,
          onToggleTTS: onToggleTTS,
        ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Markdown StyleSheet helper
// ---------------------------------------------------------------------------

MarkdownStyleSheet _markdownStyleSheet() {
  return MarkdownStyleSheet(
    p: GoogleFonts.poppins(fontSize: 14, color: Colors.white, height: 1.7),
    strong: GoogleFonts.poppins(
      fontSize: 14,
      color: Colors.white,
      fontWeight: FontWeight.w600,
    ),
    em: GoogleFonts.poppins(
      fontSize: 14,
      color: Colors.white,
      fontStyle: FontStyle.italic,
    ),
    h1: GoogleFonts.poppins(
      fontSize: 20,
      color: Colors.white,
      fontWeight: FontWeight.w700,
      height: 1.4,
    ),
    h2: GoogleFonts.poppins(
      fontSize: 17,
      color: Colors.white,
      fontWeight: FontWeight.w600,
      height: 1.4,
    ),
    h3: GoogleFonts.poppins(
      fontSize: 15,
      color: Colors.white,
      fontWeight: FontWeight.w600,
      height: 1.4,
    ),
    code: GoogleFonts.sourceCodePro(
      fontSize: 13,
      color: const Color(0xFF16DB65),
      backgroundColor: const Color(0xFF1A2A1A),
    ),
    codeblockDecoration: BoxDecoration(
      color: const Color(0xFF0A1A0A),
      borderRadius: BorderRadius.circular(8),
      border: Border.all(color: const Color(0xFF16DB65).withOpacity(0.2)),
    ),
    codeblockPadding: const EdgeInsets.all(14),
    listBullet: GoogleFonts.poppins(
      fontSize: 14,
      color: const Color(0xFF16DB65),
    ),
    listIndent: 20,
    blockquote: GoogleFonts.poppins(
      fontSize: 14,
      color: const Color(0xFFCCCCCC),
      fontStyle: FontStyle.italic,
      height: 1.6,
    ),
    blockquoteDecoration: BoxDecoration(
      border: Border(
        left: BorderSide(
          color: const Color(0xFF16DB65).withOpacity(0.5),
          width: 3,
        ),
      ),
    ),
    blockquotePadding: const EdgeInsets.only(left: 12),
    a: GoogleFonts.poppins(
      fontSize: 14,
      color: const Color(0xFF16DB65),
      decoration: TextDecoration.underline,
      decorationColor: const Color(0xFF16DB65).withOpacity(0.5),
    ),
  );
}

// ---------------------------------------------------------------------------
// Shared Bubbles
// ---------------------------------------------------------------------------

class _AiAvatar extends StatelessWidget {
  @override
  Widget build(BuildContext context) => Container(
    width: 30,
    height: 30,
    decoration: BoxDecoration(
      shape: BoxShape.circle,
      color: const Color(0x3316DB65),
      border: Border.all(color: const Color(0xFF16DB65).withOpacity(0.4)),
    ),
    child: const Icon(Icons.eco_rounded, color: Color(0xFF16DB65), size: 15),
  );
}

class _UserBubble extends StatelessWidget {
  const _UserBubble({required this.text});
  final String text;

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: Alignment.centerRight,
      child: Container(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.65,
        ),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        decoration: BoxDecoration(
          color: const Color(0x3316DB65),
          borderRadius: const BorderRadius.only(
            topLeft: Radius.circular(16),
            topRight: Radius.circular(16),
            bottomLeft: Radius.circular(16),
            bottomRight: Radius.circular(4),
          ),
          border: Border.all(color: const Color(0xFF16DB65).withOpacity(0.25)),
        ),
        child: SelectionArea(
          child: Text(
            text,
            style: GoogleFonts.poppins(
              fontSize: 14,
              color: Colors.white,
              height: 1.6,
            ),
          ),
        ),
      ),
    );
  }
}

class _AiBubble extends StatelessWidget {
  const _AiBubble({required this.text});
  final String text;

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _AiAvatar(),
        const SizedBox(width: 10),
        Expanded(
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            decoration: BoxDecoration(
              color: const Color(0xFF111111),
              borderRadius: const BorderRadius.only(
                topLeft: Radius.circular(4),
                topRight: Radius.circular(16),
                bottomLeft: Radius.circular(16),
                bottomRight: Radius.circular(16),
              ),
              border: Border.all(color: const Color(0xFF1A1A1A)),
            ),
            child: SelectionArea(
              child: MarkdownBody(
                data: text,
                selectable: false,
                extensionSet: md.ExtensionSet.gitHubWeb,
                onTapLink: (text, href, title) {
                  if (href != null) launchUrl(Uri.parse(href));
                },
                styleSheet: _markdownStyleSheet(),
              ),
            ),
          ),
        ),
      ],
    );
  }
}

class _ErrorBubble extends StatelessWidget {
  const _ErrorBubble({required this.text});
  final String text;

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          width: 30,
          height: 30,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: const Color(0x33FF4444),
            border: Border.all(color: const Color(0xFFFF4444).withOpacity(0.4)),
          ),
          child: const Icon(
            Icons.error_outline_rounded,
            color: Color(0xFFFF4444),
            size: 15,
          ),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            decoration: BoxDecoration(
              color: const Color(0xFF1A0A0A),
              borderRadius: const BorderRadius.only(
                topLeft: Radius.circular(4),
                topRight: Radius.circular(16),
                bottomLeft: Radius.circular(16),
                bottomRight: Radius.circular(16),
              ),
              border: Border.all(
                color: const Color(0xFFFF4444).withOpacity(0.3),
              ),
            ),
            child: Text(
              text.isNotEmpty
                  ? text
                  : 'Terjadi kesalahan saat memproses pertanyaan.',
              style: GoogleFonts.poppins(
                fontSize: 14,
                color: const Color(0xFFFF8888),
                height: 1.6,
              ),
            ),
          ),
        ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Intents untuk Shortcuts
// ---------------------------------------------------------------------------

class SendMessageIntent extends Intent {
  const SendMessageIntent();
}

// ---------------------------------------------------------------------------
// Input Bar
// ---------------------------------------------------------------------------

class _InputBar extends StatefulWidget {
  const _InputBar({
    required this.controller,
    required this.focusNode,
    required this.sending,
    required this.onSend,
    required this.onUploadPdfs,
    required this.onSetModel,
    required this.onGetModels,
    this.pendingDetailId,
    this.onStop,
  });

  final TextEditingController controller;
  final FocusNode focusNode;
  final bool sending;
  final VoidCallback onSend;
  final Future<void> Function(List<PdfUploadFile> files, {void Function(int done, int total)? onProgress}) onUploadPdfs;
  // Dipanggil saat user menekan Terapkan: mode + path opsional
  final Future<bool> Function(String mode, {String? path}) onSetModel;
  // Dipanggil saat dialog dibuka untuk mengambil daftar model lokal
  final Future<List<Map<String, String>>> Function() onGetModels;
  final int? pendingDetailId;
  final VoidCallback? onStop;

  @override
  State<_InputBar> createState() => _InputBarState();
}

class _InputBarState extends State<_InputBar> {
  bool _hasText = false;
  final stt.SpeechToText _speechToText = stt.SpeechToText();
  bool _isListening = false;
  bool _speechEnabled = false;

  String _llmMode = 'groq';
  String _localModelName = '';
  bool _modelSwitching = false;

  @override
  void initState() {
    super.initState();
    widget.controller.addListener(_onTextChanged);  // ← Pisahkan ke method terpisah
    _initSpeech();
  }

  // Method terpisah untuk text change listener
  void _onTextChanged() {
    final has = widget.controller.text.trim().isNotEmpty;
    if (has != _hasText && mounted) {
      setState(() => _hasText = has);
    }
  }
  
  @override
  void dispose() {
    // 1. Hapus listener dari controller
    widget.controller.removeListener(_onTextChanged);
    
    // 2. Hentikan STT jika sedang mendengarkan
    _speechToText.stop();
    
    // 3. Panggil super dispose (WAJIB)
    super.dispose();
  }
  // ================================================================

  void _initSpeech() async {
    try {
      _speechEnabled = await _speechToText.initialize(
        onStatus: (status) {
          if ((status == 'done' || status == 'notListening') && mounted) {
            setState(() => _isListening = false);
          }
        },
        onError: (error) {
          if (mounted) setState(() => _isListening = false);
        },
      );
      if (mounted) setState(() {});
    } catch (e) {
      debugPrint('STT Error: $e');
    }
  }

  void _startListening() async {
    await _speechToText.listen(
      onResult: (result) {
        if (mounted) {
          setState(() {
            widget.controller.text = result.recognizedWords;
            widget.controller.selection = TextSelection.collapsed(
                offset: widget.controller.text.length);
          });
        }
      },
      localeId: 'id_ID',
    );
    if (mounted) setState(() => _isListening = true);
  }

  void _stopListening() async {
    await _speechToText.stop();
    if (mounted) setState(() => _isListening = false);
  }

  // ── Model Selector Dialog ─────────────────────────────────────────────────

  Future<void> _showModelDialog(BuildContext context) async {
    List<Map<String, String>> _localModels = [];
    bool _loadingModels = true;
    String? _selectedPath = _llmMode == 'local' ? null : null;
    String _dialogMode = _llmMode;
    final isMobile = MediaQuery.of(context).size.width < 600;

    await showDialog(
      context: context,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setDS) {
          if (_loadingModels) {
            Future.microtask(() async {
              try {
                final models = await widget.onGetModels();
                if (ctx.mounted) {
                  setDS(() {
                    _localModels = models;
                    _loadingModels = false;
                  });
                }
              } catch (_) {
                if (ctx.mounted) setDS(() => _loadingModels = false);
              }
            });
          }

          return Dialog(
            backgroundColor: const Color(0xFF111111),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
              side: const BorderSide(color: Color(0xFF1A1A1A)),
            ),
            child: Container(
              width: isMobile ? double.infinity : 420,
              constraints: BoxConstraints(
                maxWidth: 500,
                maxHeight: isMobile 
                    ? MediaQuery.of(ctx).size.height * 0.9  // 90% height di mobile
                    : 600,
              ),
              child: Padding(
                padding: EdgeInsets.all(isMobile ? 16 : 24),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // ── Header ──────────────────────────────────────────────
                    Row(
                      children: [
                        Container(
                          width: 36, height: 36,
                          decoration: BoxDecoration(
                            color: const Color(0xFF16DB65).withOpacity(0.12),
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: const Icon(Icons.memory_rounded,
                              color: Color(0xFF16DB65), size: 18),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Text(
                            'Pilih Model LLM',
                            style: GoogleFonts.poppins(
                              fontSize: isMobile ? 14 : 15,
                              fontWeight: FontWeight.w600,
                              color: Colors.white,
                            ),
                          ),
                        ),
                        GestureDetector(
                          onTap: () => Navigator.of(ctx).pop(),
                          child: const Icon(Icons.close_rounded,
                              color: Color(0xFF666666), size: 20),
                        ),
                      ],
                    ),
                    const SizedBox(height: 6),
                    Text(
                      'Mode "Lokal" memindah embedding & reranker ke CPU',
                      style: GoogleFonts.poppins(
                          fontSize: 11, color: const Color(0xFF666666)),
                    ),
                    const SizedBox(height: 20),

                    // ── Scrollable content ──────────────────────────────────
                    Expanded(
                      child: SingleChildScrollView(
                        child: Column(
                          children: [
                            // Groq API option
                            _ModelOptionTile(
                              icon: Icons.cloud_rounded,
                              title: 'Groq API',
                              subtitle: 'llama-3.3-70b-versatile · GPU bebas untuk pipeline',
                              selected: _dialogMode == 'groq',
                              accentColor: const Color(0xFF16DB65),
                              onTap: () => setDS(() {
                                _dialogMode = 'groq';
                                _selectedPath = null;
                              }),
                            ),
                            const SizedBox(height: 10),

                            // Local model section
                            _ModelOptionTile(
                              icon: Icons.folder_special_rounded,
                              title: 'Model Lokal',
                              subtitle: 'Folder HuggingFace dari folder model/',
                              selected: _dialogMode == 'local',
                              accentColor: const Color(0xFFFFAA33),
                              onTap: () => setDS(() => _dialogMode = 'local'),
                            ),

                            // Sub-list model lokal
                            if (_dialogMode == 'local') ...[
                              const SizedBox(height: 8),
                              Container(
                                constraints: BoxConstraints(
                                  maxHeight: isMobile ? 300 : 250,
                                ),
                                decoration: BoxDecoration(
                                  color: const Color(0xFF0D0D0D),
                                  borderRadius: BorderRadius.circular(10),
                                  border: Border.all(color: const Color(0xFF1E1E1E)),
                                ),
                                child: _loadingModels
                                    ? const Center(
                                        child: Padding(
                                          padding: EdgeInsets.all(20),
                                          child: CircularProgressIndicator(
                                            color: Color(0xFF16DB65), strokeWidth: 2),
                                        ),
                                      )
                                    : _localModels.isEmpty
                                        ? Padding(
                                            padding: const EdgeInsets.all(16),
                                            child: Text(
                                              'Tidak ada model di folder model/\n'
                                              'Pastikan folder berisi config.json dan model.safetensors',
                                              style: GoogleFonts.poppins(
                                                  fontSize: 12,
                                                  color: const Color(0xFF666666)),
                                              textAlign: TextAlign.center,
                                            ),
                                          )
                                        : ListView.separated(
                                            shrinkWrap: true,
                                            physics: const BouncingScrollPhysics(),
                                            padding: const EdgeInsets.all(8),
                                            itemCount: _localModels.length,
                                            separatorBuilder: (_, __) =>
                                                const SizedBox(height: 4),
                                            itemBuilder: (_, i) {
                                              final m = _localModels[i];
                                              final isActive = _selectedPath == m['path'];
                                              return InkWell(
                                                borderRadius: BorderRadius.circular(8),
                                                onTap: () => setDS(
                                                    () => _selectedPath = m['path']),
                                                child: Container(
                                                  padding: const EdgeInsets.symmetric(
                                                      horizontal: 12, vertical: 9),
                                                  decoration: BoxDecoration(
                                                    color: isActive
                                                        ? const Color(0xFFFFAA33)
                                                            .withOpacity(0.10)
                                                        : Colors.transparent,
                                                    borderRadius: BorderRadius.circular(8),
                                                    border: isActive
                                                        ? Border.all(
                                                            color: const Color(0xFFFFAA33)
                                                                .withOpacity(0.4))
                                                        : null,
                                                  ),
                                                  child: Row(
                                                    children: [
                                                      Icon(
                                                        Icons.folder_special_rounded,
                                                        size: 15,
                                                        color: isActive
                                                            ? const Color(0xFFFFAA33)
                                                            : const Color(0xFF666666),
                                                      ),
                                                      const SizedBox(width: 10),
                                                      Expanded(
                                                        child: Column(
                                                          crossAxisAlignment:
                                                              CrossAxisAlignment.start,
                                                          children: [
                                                            Text(
                                                              m['name'] ?? '',
                                                              style: GoogleFonts.poppins(
                                                                fontSize: 12,
                                                                color: isActive
                                                                    ? const Color(0xFFFFAA33)
                                                                    : Colors.white,
                                                                fontWeight: isActive
                                                                    ? FontWeight.w600
                                                                    : FontWeight.normal,
                                                              ),
                                                              overflow: TextOverflow.ellipsis,
                                                            ),
                                                            Text(
                                                              'HuggingFace · transformers',
                                                              style: GoogleFonts.poppins(
                                                                  fontSize: 10,
                                                                  color: const Color(0xFF555555)),
                                                            ),
                                                          ],
                                                        ),
                                                      ),
                                                      if (isActive)
                                                        const Icon(
                                                            Icons.check_rounded,
                                                            size: 14,
                                                            color: Color(0xFFFFAA33)),
                                                    ],
                                                  ),
                                                ),
                                              );
                                            },
                                          ),
                              ),
                            ],
                          ],
                        ),
                      ),
                    ),

                    const SizedBox(height: 20),

                    // ── Tombol ──────────────────────────────────────────────
                    Row(
                      children: [
                        Expanded(
                          child: TextButton(
                            onPressed: () => Navigator.of(ctx).pop(),
                            style: TextButton.styleFrom(
                              padding: const EdgeInsets.symmetric(vertical: 12),
                              shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(10)),
                            ),
                            child: Text('Batal',
                                style: GoogleFonts.poppins(
                                    fontSize: 13, color: const Color(0xFF888888))),
                          ),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: ElevatedButton(
                            onPressed: (_dialogMode == 'local' &&
                                    (_selectedPath == null || _selectedPath!.isEmpty))
                                ? null
                                : () async {
                                    Navigator.of(ctx).pop();
                                    if (mounted) {
                                      setState(() => _modelSwitching = true);
                                    }
                                    try {
                                      await widget.onSetModel(
                                        _dialogMode,
                                        path: _selectedPath,
                                      );
                                      if (mounted) {
                                        setState(() {
                                          _llmMode = _dialogMode;
                                          _localModelName = _selectedPath != null
                                              ? _selectedPath!.split('/').last.split('\\').last
                                              : '';
                                          _modelSwitching = false;
                                        });
                                      }
                                    } catch (e) {
                                      if (mounted) {
                                        setState(() => _modelSwitching = false);
                                      }
                                      // Show error snackbar
                                      if (mounted) {
                                        ScaffoldMessenger.of(context).showSnackBar(
                                          SnackBar(
                                            content: Text('Gagal mengganti model: $e'),
                                            backgroundColor: Colors.red,
                                          ),
                                        );
                                      }
                                    }
                                  },
                            style: ElevatedButton.styleFrom(
                              backgroundColor: const Color(0xFF16DB65),
                              disabledBackgroundColor: const Color(0xFF1E1E1E),
                              padding: const EdgeInsets.symmetric(vertical: 12),
                              shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(10)),
                              elevation: 0,
                            ),
                            child: Text(
                              'Terapkan',
                              style: GoogleFonts.poppins(
                                fontSize: 13,
                                fontWeight: FontWeight.w600,
                                color: (_dialogMode == 'local' &&
                                        (_selectedPath == null || _selectedPath!.isEmpty))
                                    ? const Color(0xFF555555)
                                    : Colors.black,
                              ),
                            ),
                          ),
                        )
                      ],
                    ),
                  ],
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  void _showUploadDialog(BuildContext context) {
    // State multi-file
    List<PdfUploadFile> _selectedFiles = [];
    bool  _isUploading = false;
    int   _uploadDone  = 0;

    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setDialogState) {

          // ── Pilih file (multi-select, Ctrl+klik / Shift+klik di desktop) ──
          Future<void> pickFiles() async {
            final result = await FilePicker.platform.pickFiles(
              type: FileType.custom,
              allowedExtensions: ['pdf'],
              allowMultiple: true,   // ← multi-select aktif
              withData: true,
            );
            if (result == null || result.files.isEmpty) return;

            final newFiles = result.files
                .where((f) => f.bytes != null)
                .map((f) => PdfUploadFile(bytes: f.bytes!, name: f.name))
                .toList();

            setDialogState(() {
              // Gabungkan, hindari nama duplikat
              final existingNames = _selectedFiles.map((f) => f.name).toSet();
              for (final f in newFiles) {
                if (!existingNames.contains(f.name)) {
                  _selectedFiles.add(f);
                  existingNames.add(f.name);
                }
              }
            });
          }

          // ── Hapus satu file dari antrian ──────────────────────────────────
          void removeFile(int index) {
            setDialogState(() => _selectedFiles.removeAt(index));
          }

          // ── Upload semua file satu per satu ───────────────────────────────
          Future<void> doUpload() async {
            if (_selectedFiles.isEmpty) return;
            setDialogState(() {
              _isUploading = true;
              _uploadDone  = 0;
            });

            await widget.onUploadPdfs(
              _selectedFiles,
              onProgress: (done, total) {
                if (ctx.mounted) {
                  setDialogState(() => _uploadDone = done);
                }
              },
            );

            if (ctx.mounted) Navigator.of(ctx).pop();
          }

          final hasFiles = _selectedFiles.isNotEmpty;

          return Dialog(
            backgroundColor: const Color(0xFF111111),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
              side: const BorderSide(color: Color(0xFF1A1A1A)),
            ),
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 480, maxHeight: 600),
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [

                    // ── Header ─────────────────────────────────────────────
                    Row(
                      children: [
                        Container(
                          width: 36, height: 36,
                          decoration: BoxDecoration(
                            color: const Color(0xFF16DB65).withOpacity(0.12),
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: const Icon(Icons.auto_stories_rounded,
                              color: Color(0xFF16DB65), size: 18),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Text(
                            'Tambah Pengetahuan Bot',
                            style: GoogleFonts.poppins(
                              fontSize: 15,
                              fontWeight: FontWeight.w600,
                              color: Colors.white,
                            ),
                          ),
                        ),
                        if (!_isUploading)
                          GestureDetector(
                            onTap: () => Navigator.of(ctx).pop(),
                            child: const Icon(Icons.close_rounded,
                                color: Color(0xFF666666), size: 20),
                          ),
                      ],
                    ),
                    const SizedBox(height: 6),
                    Text(
                      'Pilih satu atau beberapa file PDF sekaligus',
                      style: GoogleFonts.poppins(
                        fontSize: 12,
                        color: const Color(0xFFA3A3A3),
                      ),
                    ),
                    const SizedBox(height: 16),

                    // ── Drop zone / tombol pilih ───────────────────────────
                    GestureDetector(
                      onTap: _isUploading ? null : pickFiles,
                      child: Container(
                        width: double.infinity,
                        padding: const EdgeInsets.symmetric(
                            vertical: 20, horizontal: 16),
                        decoration: BoxDecoration(
                          color: const Color(0xFF0D0D0D),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(
                            color: hasFiles
                                ? const Color(0xFF16DB65).withOpacity(0.4)
                                : const Color(0xFF2A2A2A),
                            width: hasFiles ? 1.5 : 1,
                          ),
                        ),
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(
                              hasFiles
                                  ? Icons.picture_as_pdf_rounded
                                  : Icons.upload_file_rounded,
                              color: hasFiles
                                  ? const Color(0xFF16DB65)
                                  : const Color(0xFF555555),
                              size: 32,
                            ),
                            const SizedBox(height: 8),
                            Text(
                              hasFiles
                                  ? 'Ketuk untuk menambah lebih banyak file'
                                  : 'Ketuk untuk memilih file PDF',
                              style: GoogleFonts.poppins(
                                fontSize: 13,
                                fontWeight: FontWeight.w500,
                                color: hasFiles
                                    ? const Color(0xFF16DB65)
                                    : const Color(0xFF888888),
                              ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              kIsWeb
                                  ? 'Tahan Ctrl untuk pilih beberapa file'
                                  : 'Hanya file .pdf yang didukung',
                              style: GoogleFonts.poppins(
                                fontSize: 11,
                                color: const Color(0xFF555555),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),

                    // ── Daftar file yang dipilih ───────────────────────────
                    if (hasFiles) ...[
                      const SizedBox(height: 12),
                      // Progress upload saat sedang upload
                      if (_isUploading) ...[
                        Padding(
                          padding: const EdgeInsets.only(bottom: 8),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Mengunggah $_uploadDone dari ${_selectedFiles.length} file...',
                                style: GoogleFonts.poppins(
                                  fontSize: 12,
                                  color: const Color(0xFF16DB65),
                                ),
                              ),
                              const SizedBox(height: 6),
                              LinearProgressIndicator(
                                value: _selectedFiles.isEmpty
                                    ? 0
                                    : _uploadDone / _selectedFiles.length,
                                backgroundColor: const Color(0xFF1A1A1A),
                                color: const Color(0xFF16DB65),
                                borderRadius: BorderRadius.circular(4),
                              ),
                            ],
                          ),
                        ),
                      ],
                      // List file
                      ConstrainedBox(
                        constraints: const BoxConstraints(maxHeight: 200),
                        child: ListView.separated(
                          shrinkWrap: true,
                          itemCount: _selectedFiles.length,
                          separatorBuilder: (_, __) =>
                              const SizedBox(height: 6),
                          itemBuilder: (_, i) {
                            final f = _selectedFiles[i];
                            final isDone = _isUploading && i < _uploadDone;
                            return Container(
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 12, vertical: 8),
                              decoration: BoxDecoration(
                                color: const Color(0xFF0D0D0D),
                                borderRadius: BorderRadius.circular(8),
                                border: Border.all(
                                  color: isDone
                                      ? const Color(0xFF16DB65).withOpacity(0.4)
                                      : const Color(0xFF1E1E1E),
                                ),
                              ),
                              child: Row(
                                children: [
                                  Icon(
                                    isDone
                                        ? Icons.check_circle_outline_rounded
                                        : Icons.picture_as_pdf_rounded,
                                    size: 16,
                                    color: isDone
                                        ? const Color(0xFF16DB65)
                                        : const Color(0xFF888888),
                                  ),
                                  const SizedBox(width: 8),
                                  Expanded(
                                    child: Text(
                                      f.name,
                                      style: GoogleFonts.poppins(
                                        fontSize: 12,
                                        color: isDone
                                            ? const Color(0xFF16DB65)
                                            : Colors.white,
                                      ),
                                      overflow: TextOverflow.ellipsis,
                                    ),
                                  ),
                                  const SizedBox(width: 4),
                                  Text(
                                    '${(f.bytes.length / 1024).toStringAsFixed(0)} KB',
                                    style: GoogleFonts.poppins(
                                      fontSize: 11,
                                      color: const Color(0xFF555555),
                                    ),
                                  ),
                                  if (!_isUploading) ...[
                                    const SizedBox(width: 6),
                                    GestureDetector(
                                      onTap: () => removeFile(i),
                                      child: const Icon(
                                        Icons.close_rounded,
                                        size: 14,
                                        color: Color(0xFF555555),
                                      ),
                                    ),
                                  ],
                                ],
                              ),
                            );
                          },
                        ),
                      ),
                    ],

                    const SizedBox(height: 16),

                    // ── Tombol Batal / Unggah ──────────────────────────────
                    Row(
                      children: [
                        Expanded(
                          child: TextButton(
                            onPressed: _isUploading
                                ? null
                                : () => Navigator.of(ctx).pop(),
                            style: TextButton.styleFrom(
                              padding:
                                  const EdgeInsets.symmetric(vertical: 12),
                              shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(10)),
                            ),
                            child: Text(
                              'Batal',
                              style: GoogleFonts.poppins(
                                fontSize: 13,
                                color: _isUploading
                                    ? const Color(0xFF444444)
                                    : const Color(0xFF888888),
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: ElevatedButton(
                            onPressed: (hasFiles && !_isUploading)
                                ? doUpload
                                : null,
                            style: ElevatedButton.styleFrom(
                              backgroundColor: (hasFiles && !_isUploading)
                                  ? const Color(0xFF16DB65)
                                  : const Color(0xFF1A1A1A),
                              padding:
                                  const EdgeInsets.symmetric(vertical: 12),
                              shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(10)),
                              elevation: 0,
                            ),
                            child: _isUploading
                                ? const SizedBox(
                                    width: 16, height: 16,
                                    child: CircularProgressIndicator(
                                      strokeWidth: 2,
                                      color: Colors.black,
                                    ),
                                  )
                                : Text(
                                    hasFiles
                                        ? 'Unggah (${_selectedFiles.length})'
                                        : 'Unggah',
                                    style: GoogleFonts.poppins(
                                      fontSize: 13,
                                      fontWeight: FontWeight.w600,
                                      color: (hasFiles && !_isUploading)
                                          ? Colors.black
                                          : const Color(0xFF555555),
                                    ),
                                  ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final hasPending = widget.pendingDetailId != null;

    return Container(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 20),
      decoration: const BoxDecoration(
        color: Color(0xFF0D0D0D),
        border: Border(top: BorderSide(color: Color(0xFF1A1A1A))),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          // Tombol Upload PDF
          Padding(
            padding: const EdgeInsets.only(bottom: 0),
            child: SizedBox(
              width: 48,
              height: 48,
              child: Tooltip(
                message: 'Unggah PDF pengetahuan bot',
                child: ElevatedButton(
                  onPressed: () => _showUploadDialog(context),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF1A1A1A),
                    padding: EdgeInsets.zero,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    elevation: 0,
                  ),
                  child: const Icon(
                    Icons.upload_file_rounded,
                    color: Color(0xFF888888),
                    size: 20,
                  ),
                ),
              ),
            ),
          ),
          const SizedBox(width: 8),

          // ── Tombol Pilih Model LLM ──────────────────────────────────────
          SizedBox(
            height: 48,
            child: Tooltip(
              message: _llmMode == 'local'
                  ? 'Model: $_localModelName'
                  : 'Model: Groq API',
              child: ElevatedButton(
                onPressed: _modelSwitching
                    ? null
                    : () => _showModelDialog(context),
                style: ElevatedButton.styleFrom(
                  backgroundColor: _llmMode == 'local'
                      ? const Color(0xFFFFAA33).withOpacity(0.12)
                      : const Color(0xFF1A1A1A),
                  padding: const EdgeInsets.symmetric(horizontal: 10),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                    side: BorderSide(
                      color: _llmMode == 'local'
                          ? const Color(0xFFFFAA33).withOpacity(0.4)
                          : Colors.transparent,
                    ),
                  ),
                  elevation: 0,
                ),
                child: _modelSwitching
                    ? const SizedBox(
                        width: 14, height: 14,
                        child: CircularProgressIndicator(
                          strokeWidth: 2, color: Color(0xFF16DB65)))
                    : Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            _llmMode == 'local'
                                ? Icons.memory_rounded
                                : Icons.cloud_rounded,
                            size: 15,
                            color: _llmMode == 'local'
                                ? const Color(0xFFFFAA33)
                                : const Color(0xFF888888),
                          ),
                          const SizedBox(width: 5),
                          Text(
                            _llmMode == 'local' ? 'Lokal' : 'Groq',
                            style: GoogleFonts.poppins(
                              fontSize: 12,
                              fontWeight: FontWeight.w500,
                              color: _llmMode == 'local'
                                  ? const Color(0xFFFFAA33)
                                  : const Color(0xFF888888),
                            ),
                          ),
                        ],
                      ),
              ),
            ),
          ),
          const SizedBox(width: 8),

          Expanded(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxHeight: 150),
              // PERBAIKAN: HANYA MENCEGAT ENTER, BIARKAN SHIFT+ENTER LEWAT SECARA DEFAULT
              child: Shortcuts(
                shortcuts: <LogicalKeySet, Intent>{
                  LogicalKeySet(LogicalKeyboardKey.enter): const SendMessageIntent(),
                  LogicalKeySet(LogicalKeyboardKey.numpadEnter): const SendMessageIntent(),
                },
                child: Actions(
                  actions: <Type, Action<Intent>>{
                    SendMessageIntent: CallbackAction<SendMessageIntent>(
                      onInvoke: (intent) {
                        if (!widget.sending && widget.controller.text.trim().isNotEmpty) {
                          widget.onSend();
                        }
                        return null;
                      },
                    ),
                  },
                  child: TextField(
                    controller: widget.controller,
                    focusNode: widget.focusNode,
                    maxLines: null,
                    keyboardType: TextInputType.multiline,
                    textInputAction: TextInputAction.newline,
                    enabled: !widget.sending,
                    style: GoogleFonts.poppins(fontSize: 14, color: Colors.white),
                    cursorColor: const Color(0xFF16DB65),
                    decoration: InputDecoration(
                      hintText: 'Ketik pertanyaan Anda...',
                      hintStyle: GoogleFonts.poppins(
                        fontSize: 14,
                        color: const Color(0xFFA3A3A3),
                      ),
                      filled: true,
                      fillColor: const Color(0xFF111111),
                      contentPadding: const EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 12,
                      ),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(14),
                        borderSide: const BorderSide(color: Color(0xFF1A1A1A)),
                      ),
                      enabledBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(14),
                        borderSide: const BorderSide(color: Color(0xFF1A1A1A)),
                      ),
                      focusedBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(14),
                        borderSide: const BorderSide(
                          color: Color(0xFF16DB65),
                          width: 1.5,
                        ),
                      ),
                      disabledBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(14),
                        borderSide: const BorderSide(color: Color(0xFF1A1A1A)),
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),
          const SizedBox(width: 10),

          // Tombol Mic STT
          Padding(
            padding: const EdgeInsets.only(bottom: 0),
            child: SizedBox(
              width: 48,
              height: 48,
              child: Tooltip(
                message: _isListening ? 'Berhenti Merekam' : 'Input Suara',
                child: ElevatedButton(
                  onPressed: _speechEnabled
                      ? (_isListening ? _stopListening : _startListening)
                      : null,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _isListening
                        ? Colors.red.shade400
                        : const Color(0xFF1A1A1A),
                    padding: EdgeInsets.zero,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    elevation: 0,
                  ),
                  child: Icon(
                    _isListening ? Icons.mic_off_rounded : Icons.mic_none_rounded,
                    color: _isListening ? Colors.white : const Color(0xFF888888),
                    size: 20,
                  ),
                ),
              ),
            ),
          ),
          const SizedBox(width: 10),

          SizedBox(
            width: 48,
            height: 48,
            child: hasPending
                ? Tooltip(
                    message: 'Hentikan generate',
                    child: ElevatedButton(
                      onPressed: widget.onStop,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFFFF4444),
                        padding: EdgeInsets.zero,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                        elevation: 0,
                      ),
                      child: const Icon(
                        Icons.stop_rounded,
                        color: Colors.white,
                        size: 20,
                      ),
                    ),
                  )
                : ElevatedButton(
                    onPressed: (widget.sending || !_hasText)
                        ? null
                        : widget.onSend,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: _hasText && !widget.sending
                          ? const Color(0xFF16DB65)
                          : const Color(0xFF1A1A1A),
                      padding: EdgeInsets.zero,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                      elevation: 0,
                    ),
                    child: widget.sending
                        ? const SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: Colors.black,
                            ),
                          )
                        : const Icon(
                            Icons.arrow_upward_rounded,
                            color: Colors.black,
                            size: 20,
                          ),
                  ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Model Option Tile — dipakai di dialog pemilihan model
// ---------------------------------------------------------------------------

class _ModelOptionTile extends StatelessWidget {
  const _ModelOptionTile({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.selected,
    required this.accentColor,
    required this.onTap,
  });

  final IconData icon;
  final String title;
  final String subtitle;
  final bool selected;
  final Color accentColor;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      borderRadius: BorderRadius.circular(12),
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 150),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
        decoration: BoxDecoration(
          color: selected
              ? accentColor.withOpacity(0.08)
              : const Color(0xFF151515),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: selected
                ? accentColor.withOpacity(0.5)
                : const Color(0xFF222222),
            width: selected ? 1.5 : 1,
          ),
        ),
        child: Row(
          children: [
            Container(
              width: 34, height: 34,
              decoration: BoxDecoration(
                color: selected
                    ? accentColor.withOpacity(0.15)
                    : const Color(0xFF1A1A1A),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Icon(icon,
                  size: 17,
                  color: selected ? accentColor : const Color(0xFF666666)),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: GoogleFonts.poppins(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      color: selected ? Colors.white : const Color(0xFFAAAAAA),
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    subtitle,
                    style: GoogleFonts.poppins(
                        fontSize: 10.5,
                        color: const Color(0xFF555555)),
                  ),
                ],
              ),
            ),
            if (selected)
              Container(
                width: 18, height: 18,
                decoration: BoxDecoration(
                  color: accentColor,
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.check_rounded,
                    size: 12, color: Colors.black),
              ),
          ],
        ),
      ),
    );
  }
}