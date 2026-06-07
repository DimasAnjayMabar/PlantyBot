// lib/chats/input_bar.dart
import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:frontend/chats/service_chats.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:file_picker/file_picker.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:image_picker/image_picker.dart';

// ---------------------------------------------------------------------------
// Intents untuk Shortcuts
// ---------------------------------------------------------------------------

class SendMessageIntent extends Intent {
  const SendMessageIntent();
}

// ---------------------------------------------------------------------------
// Input Bar
// ---------------------------------------------------------------------------

class InputBar extends StatefulWidget {
  const InputBar({
    super.key,
    required this.controller,
    required this.focusNode,
    required this.sending,
    required this.onSend,
    required this.onUploadPdfs,
    required this.onSetModel,
    required this.onGetModels,
    required this.onSetRagMode,
    required this.onGetRagMode,
    this.pendingDetailId,
    this.onStop,
  });

  final TextEditingController controller;
  final FocusNode focusNode;
  final bool sending;
  final void Function({Uint8List? imageBytes, String? imageName}) onSend;
  final Future<void> Function(
    List<PdfUploadFile> files, {
    void Function(int done, int total)? onProgress,
  }) onUploadPdfs;
  final Future<bool> Function(String mode, {String? path}) onSetModel;
  final Future<List<Map<String, dynamic>>> Function() onGetModels;
  final Future<bool> Function(String mode) onSetRagMode;
  final Future<String> Function() onGetRagMode;
  final int? pendingDetailId;
  final VoidCallback? onStop;

  @override
  State<InputBar> createState() => _InputBarState();
}

class _InputBarState extends State<InputBar> {
  bool _hasText = false;
  final stt.SpeechToText _speechToText = stt.SpeechToText();
  bool _isListening = false;
  bool _speechEnabled = false;

  String _activeModelId = 'llama-3.3-70b-versatile';
  bool _modelSwitching = false;

  String _ragMode = 'improved';  
  bool _ragModeSwitching = false;

  // Image Picker State
  Uint8List? _selectedImageBytes;
  String? _selectedImageName;
  final ImagePicker _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    widget.controller.addListener(_onTextChanged);
    _initSpeech();
    _initRagMode();
  }

  void _onTextChanged() {
    final has = widget.controller.text.trim().isNotEmpty;
    if (has != _hasText && mounted) {
      setState(() => _hasText = has);
    }
  }

  @override
  void dispose() {
    widget.controller.removeListener(_onTextChanged);
    _speechToText.stop();
    super.dispose();
  }

  Future<void> _initRagMode() async {
    try {
      final mode = await widget.onGetRagMode();
      if (mounted) {
        setState(() => _ragMode = mode);
      }
    } catch (e) {
      debugPrint('Error loading RAG mode: $e');
    }
  }

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
              offset: widget.controller.text.length,
            );
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

  // Pick Image Logic
  Future<void> _pickImage() async {
    try {
      final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
      if (image != null) {
        final bytes = await image.readAsBytes();
        setState(() {
          _selectedImageBytes = bytes;
          _selectedImageName = image.name;
        });
      }
    } catch (e) {
      debugPrint('Image picker error: $e');
    }
  }

  void _triggerSend() {
    widget.onSend(
      imageBytes: _selectedImageBytes, 
      imageName: _selectedImageName
    );
    setState(() {
      _selectedImageBytes = null;
      _selectedImageName = null;
    });
  }

  // ── Dialog RAG Mode ────────────────────────────────────────
  void _showRagModeDialog(BuildContext context) {
    final isMobile = MediaQuery.of(context).size.width < 600;
    String selectedMode = _ragMode;

    showDialog(
      context: context,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setDS) => Dialog(
          backgroundColor: const Color(0xFF111111),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
            side: const BorderSide(color: Color(0xFF1A1A1A)),
          ),
          child: Container(
            width: isMobile ? double.infinity : 360,
            padding: EdgeInsets.all(isMobile ? 20 : 24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Container(
                      width: 36,
                      height: 36,
                      decoration: BoxDecoration(
                        color: const Color(0xFF16DB65).withOpacity(0.12),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: const Icon(
                        Icons.tune_rounded,
                        color: Color(0xFF16DB65),
                        size: 18,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        'Mode RAG',
                        style: GoogleFonts.poppins(
                          fontSize: 15,
                          fontWeight: FontWeight.w600,
                          color: Colors.white,
                        ),
                      ),
                    ),
                    GestureDetector(
                      onTap: () => Navigator.of(ctx).pop(),
                      child: const Icon(
                        Icons.close_rounded,
                        color: Color(0xFF666666),
                        size: 20,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 6),
                Text(
                  'Pilih metode retrieval untuk menjawab pertanyaan teknis',
                  style: GoogleFonts.poppins(
                    fontSize: 11,
                    color: const Color(0xFF888888),
                  ),
                ),
                const SizedBox(height: 16),
                
                _buildRagModeOption(
                  mode: 'improved',
                  title: 'Improved RAG',
                  description: 'Lebih akurat untuk pertanyaan kompleks',
                  icon: Icons.auto_awesome_rounded,
                  selectedMode: selectedMode,
                  onSelect: () => setDS(() => selectedMode = 'improved'),
                ),
                const SizedBox(height: 8),
                
                _buildRagModeOption(
                  mode: 'regular',
                  title: 'Regular RAG',
                  description: 'Lebih cepat, cocok untuk pertanyaan langsung',
                  icon: Icons.speed_rounded,
                  selectedMode: selectedMode,
                  onSelect: () => setDS(() => selectedMode = 'regular'),
                ),
                
                const SizedBox(height: 20),
                Row(
                  children: [
                    Expanded(
                      child: TextButton(
                        onPressed: () => Navigator.of(ctx).pop(),
                        style: TextButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 12),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(10),
                          ),
                        ),
                        child: Text(
                          'Batal',
                          style: GoogleFonts.poppins(
                            fontSize: 13,
                            color: const Color(0xFF888888),
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: ElevatedButton(
                        onPressed: selectedMode == _ragMode
                            ? null
                            : () async {
                                Navigator.of(ctx).pop();
                                if (mounted) {
                                  setState(() => _ragModeSwitching = true);
                                }
                                try {
                                  final success = await widget.onSetRagMode(selectedMode);
                                  if (mounted && success) {
                                    setState(() {
                                      _ragMode = selectedMode;
                                      _ragModeSwitching = false;
                                    });
                                    ScaffoldMessenger.of(context).showSnackBar(
                                      SnackBar(
                                        content: Text(
                                          selectedMode == 'improved'
                                              ? 'Mode RAG: Improved (Neo4j + context)'
                                              : 'Mode RAG: Regular (Simple retrieval)',
                                        ),
                                        backgroundColor: const Color(0xFF16DB65),
                                        duration: const Duration(seconds: 2),
                                      ),
                                    );
                                  } else if (mounted) {
                                    setState(() => _ragModeSwitching = false);
                                    ScaffoldMessenger.of(context).showSnackBar(
                                      const SnackBar(
                                        content: Text('Gagal mengganti mode RAG'),
                                        backgroundColor: Colors.red,
                                      ),
                                    );
                                  }
                                } catch (e) {
                                  if (mounted) {
                                    setState(() => _ragModeSwitching = false);
                                    ScaffoldMessenger.of(context).showSnackBar(
                                      SnackBar(
                                        content: Text('Error: $e'),
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
                            borderRadius: BorderRadius.circular(10),
                          ),
                          elevation: 0,
                        ),
                        child: Text(
                          'Terapkan',
                          style: GoogleFonts.poppins(
                            fontSize: 13,
                            fontWeight: FontWeight.w600,
                            color: selectedMode == _ragMode
                                ? const Color(0xFF555555)
                                : Colors.black,
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
      ),
    );
  }

  Widget _buildRagModeOption({
    required String mode,
    required String title,
    required String description,
    required IconData icon,
    required String selectedMode,
    required VoidCallback onSelect,
  }) {
    final isSelected = selectedMode == mode;
    final Color accent = mode == 'improved' 
        ? const Color(0xFF16DB65) 
        : const Color(0xFFFFAA33);

    return InkWell(
      borderRadius: BorderRadius.circular(12),
      onTap: onSelect,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 150),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
        decoration: BoxDecoration(
          color: isSelected ? accent.withOpacity(0.08) : const Color(0xFF151515),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: isSelected ? accent.withOpacity(0.5) : const Color(0xFF222222),
            width: isSelected ? 1.5 : 1,
          ),
        ),
        child: Row(
          children: [
            Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: isSelected ? accent.withOpacity(0.15) : const Color(0xFF1A1A1A),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Icon(
                icon,
                size: 20,
                color: isSelected ? accent : const Color(0xFF666666),
              ),
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
                      color: isSelected ? Colors.white : const Color(0xFFAAAAAA),
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    description,
                    style: GoogleFonts.poppins(
                      fontSize: 10.5,
                      color: const Color(0xFF555555),
                    ),
                  ),
                ],
              ),
            ),
            if (isSelected)
              Container(
                width: 18,
                height: 18,
                decoration: BoxDecoration(
                  color: accent,
                  shape: BoxShape.circle,
                ),
                child: const Icon(
                  Icons.check_rounded,
                  size: 12,
                  color: Colors.black,
                ),
              ),
          ],
        ),
      ),
    );
  }

  // ── Model Selector Dialog ─────────────────────────────────────────────────

  Future<void> _showModelDialog(BuildContext context) async {
    List<Map<String, dynamic>> _groqModels = [];
    bool _loadingModels = true;
    String _selectedId = _activeModelId;
    final isMobile = MediaQuery.of(context).size.width < 600;

    await showDialog(
      context: context,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setDS) {
          if (_loadingModels) {
            Future.microtask(() async {
              try {
                final raw = await widget.onGetModels();
                if (ctx.mounted) {
                  setDS(() {
                    _groqModels = raw.map((m) => Map<String, dynamic>.from(m)).toList();
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
                maxHeight: isMobile ? MediaQuery.of(ctx).size.height * 0.9 : 620,
              ),
              child: Padding(
                padding: EdgeInsets.all(isMobile ? 16 : 24),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Container(
                          width: 36,
                          height: 36,
                          decoration: BoxDecoration(
                            color: const Color(0xFF16DB65).withOpacity(0.12),
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: const Icon(
                            Icons.auto_awesome_rounded,
                            color: Color(0xFF16DB65),
                            size: 18,
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Pilih Model',
                                style: GoogleFonts.poppins(
                                  fontSize: isMobile ? 14 : 15,
                                  fontWeight: FontWeight.w600,
                                  color: Colors.white,
                                ),
                              ),
                              Text(
                                'Semua model berjalan via Groq API',
                                style: GoogleFonts.poppins(
                                  fontSize: 10.5,
                                  color: const Color(0xFF555555),
                                ),
                              ),
                            ],
                          ),
                        ),
                        GestureDetector(
                          onTap: () => Navigator.of(ctx).pop(),
                          child: const Icon(
                            Icons.close_rounded,
                            color: Color(0xFF666666),
                            size: 20,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                    Expanded(
                      child: _loadingModels
                          ? const Center(
                              child: CircularProgressIndicator(
                                color: Color(0xFF16DB65),
                                strokeWidth: 2,
                              ),
                            )
                          : _groqModels.isEmpty
                          ? Center(
                              child: Text(
                                'Gagal memuat daftar model.',
                                style: GoogleFonts.poppins(
                                  fontSize: 12,
                                  color: const Color(0xFF666666),
                                ),
                              ),
                            )
                          : ListView.separated(
                              shrinkWrap: true,
                              physics: const BouncingScrollPhysics(),
                              itemCount: _groqModels.length,
                              separatorBuilder: (_, __) => const SizedBox(height: 8),
                              itemBuilder: (_, i) {
                                final m = _groqModels[i];
                                final id = m['id'] as String;
                                final name = m['name'] as String;
                                final provider = m['provider'] as String;
                                final desc = m['description'] as String? ?? '';
                                final tier = m['tier'] as String? ?? 'large';
                                final isSelected = _selectedId == id;

                                final Color accent = id.startsWith('mistral')
                                    ? const Color(0xFF6C8EFF)
                                    : id.startsWith('qwen')
                                    ? const Color(0xFFFF8C42)
                                    : const Color(0xFF16DB65);

                                final String tierLabel = tier == 'small'
                                    ? 'Ringan'
                                    : tier == 'medium'
                                    ? 'Sedang'
                                    : 'Kuat';
                                final Color tierColor = tier == 'small'
                                    ? const Color(0xFF16DB65)
                                    : tier == 'medium'
                                    ? const Color(0xFFFFAA33)
                                    : const Color(0xFFFF6B6B);

                                return InkWell(
                                  borderRadius: BorderRadius.circular(12),
                                  onTap: () => setDS(() => _selectedId = id),
                                  child: AnimatedContainer(
                                    duration: const Duration(milliseconds: 150),
                                    padding: const EdgeInsets.symmetric(
                                      horizontal: 14,
                                      vertical: 12,
                                    ),
                                    decoration: BoxDecoration(
                                      color: isSelected
                                          ? accent.withOpacity(0.08)
                                          : const Color(0xFF151515),
                                      borderRadius: BorderRadius.circular(12),
                                      border: Border.all(
                                        color: isSelected
                                            ? accent.withOpacity(0.5)
                                            : const Color(0xFF222222),
                                        width: isSelected ? 1.5 : 1,
                                      ),
                                    ),
                                    child: Row(
                                      children: [
                                        Container(
                                          width: 34,
                                          height: 34,
                                          decoration: BoxDecoration(
                                            color: isSelected
                                                ? accent.withOpacity(0.15)
                                                : const Color(0xFF1A1A1A),
                                            borderRadius: BorderRadius.circular(8),
                                          ),
                                          child: Icon(
                                            Icons.cloud_rounded,
                                            size: 17,
                                            color: isSelected
                                                ? accent
                                                : const Color(0xFF666666),
                                          ),
                                        ),
                                        const SizedBox(width: 12),
                                        Expanded(
                                          child: Column(
                                            crossAxisAlignment: CrossAxisAlignment.start,
                                            children: [
                                              Row(
                                                children: [
                                                  Expanded(
                                                    child: Text(
                                                      name,
                                                      style: GoogleFonts.poppins(
                                                        fontSize: 13,
                                                        fontWeight: FontWeight.w600,
                                                        color: isSelected
                                                            ? Colors.white
                                                            : const Color(0xFFAAAAAA),
                                                      ),
                                                    ),
                                                  ),
                                                  Container(
                                                    padding: const EdgeInsets.symmetric(
                                                      horizontal: 6,
                                                      vertical: 2,
                                                    ),
                                                    decoration: BoxDecoration(
                                                      color: tierColor.withOpacity(0.12),
                                                      borderRadius: BorderRadius.circular(4),
                                                    ),
                                                    child: Text(
                                                      tierLabel,
                                                      style: GoogleFonts.poppins(
                                                        fontSize: 9.5,
                                                        color: tierColor,
                                                        fontWeight: FontWeight.w600,
                                                      ),
                                                    ),
                                                  ),
                                                ],
                                              ),
                                              const SizedBox(height: 2),
                                              Text(
                                                provider,
                                                style: GoogleFonts.poppins(
                                                  fontSize: 10.5,
                                                  color: const Color(0xFF555555),
                                                ),
                                              ),
                                              if (desc.isNotEmpty) ...[
                                                const SizedBox(height: 3),
                                                Text(
                                                  desc,
                                                  style: GoogleFonts.poppins(
                                                    fontSize: 10,
                                                    color: const Color(0xFF444444),
                                                  ),
                                                  maxLines: 2,
                                                  overflow: TextOverflow.ellipsis,
                                                ),
                                              ],
                                            ],
                                          ),
                                        ),
                                        const SizedBox(width: 8),
                                        if (isSelected)
                                          Container(
                                            width: 18,
                                            height: 18,
                                            decoration: BoxDecoration(
                                              color: accent,
                                              shape: BoxShape.circle,
                                            ),
                                            child: const Icon(
                                              Icons.check_rounded,
                                              size: 12,
                                              color: Colors.black,
                                            ),
                                          ),
                                      ],
                                    ),
                                  ),
                                );
                              },
                            ),
                    ),
                    const SizedBox(height: 20),
                    Row(
                      children: [
                        Expanded(
                          child: TextButton(
                            onPressed: () => Navigator.of(ctx).pop(),
                            style: TextButton.styleFrom(
                              padding: const EdgeInsets.symmetric(vertical: 12),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(10),
                              ),
                            ),
                            child: Text(
                              'Batal',
                              style: GoogleFonts.poppins(
                                fontSize: 13,
                                color: const Color(0xFF888888),
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: ElevatedButton(
                            onPressed: _selectedId == _activeModelId
                                ? null
                                : () async {
                                    Navigator.of(ctx).pop();
                                    if (mounted) {
                                      setState(() => _modelSwitching = true);
                                    }
                                    try {
                                      await widget.onSetModel(
                                        'groq',
                                        path: _selectedId,
                                      );
                                      if (mounted) {
                                        setState(() {
                                          _activeModelId = _selectedId;
                                          _modelSwitching = false;
                                        });
                                      }
                                    } catch (e) {
                                      if (mounted) {
                                        setState(() => _modelSwitching = false);
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
                                borderRadius: BorderRadius.circular(10),
                              ),
                              elevation: 0,
                            ),
                            child: Text(
                              'Terapkan',
                              style: GoogleFonts.poppins(
                                fontSize: 13,
                                fontWeight: FontWeight.w600,
                                color: _selectedId == _activeModelId
                                    ? const Color(0xFF555555)
                                    : Colors.black,
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

  void _showUploadDialog(BuildContext context) {
    List<PdfUploadFile> _selectedFiles = [];
    bool _isUploading = false;
    int _uploadDone = 0;
    String _embedderType = 'improved'; 

    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => StatefulBuilder(
        builder: (ctx, setDialogState) {
          Future<void> pickFiles() async {
            final result = await FilePicker.platform.pickFiles(
              type: FileType.custom,
              allowedExtensions: ['pdf'],
              allowMultiple: true,
              withData: true,
            );
            if (result == null || result.files.isEmpty) return;

            final newFiles = result.files
                .where((f) => f.bytes != null)
                .map((f) => PdfUploadFile(
                      bytes: f.bytes!,
                      name: f.name,
                      embedderType: _embedderType,
                    ))
                .toList();

            setDialogState(() {
              final existingNames = _selectedFiles.map((f) => f.name).toSet();
              for (final f in newFiles) {
                if (!existingNames.contains(f.name)) {
                  _selectedFiles.add(f);
                  existingNames.add(f.name);
                }
              }
            });
          }

          void removeFile(int index) {
            setDialogState(() => _selectedFiles.removeAt(index));
          }

          Future<void> doUpload() async {
            if (_selectedFiles.isEmpty) return;
            setDialogState(() {
              _isUploading = true;
              _uploadDone = 0;
            });

            final filesToUpload = _selectedFiles
                .map((f) => PdfUploadFile(
                      bytes: f.bytes,
                      name: f.name,
                      judul: f.judul,
                      penulis: f.penulis,
                      tahun: f.tahun,
                      embedderType: _embedderType,
                    ))
                .toList();

            await widget.onUploadPdfs(
              filesToUpload,
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
                    Row(
                      children: [
                        Container(
                          width: 36,
                          height: 36,
                          decoration: BoxDecoration(
                            color: const Color(0xFF16DB65).withOpacity(0.12),
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: const Icon(
                            Icons.auto_stories_rounded,
                            color: Color(0xFF16DB65),
                            size: 18,
                          ),
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
                            child: const Icon(
                              Icons.close_rounded,
                              color: Color(0xFF666666),
                              size: 20,
                            ),
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
                    const SizedBox(height: 12),

                    // ── Pilihan Embedder ──────────────────────────────────
                    Container(
                      decoration: BoxDecoration(
                        color: const Color(0xFF0D0D0D),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: const Color(0xFF2A2A2A)),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Padding(
                            padding: const EdgeInsets.fromLTRB(14, 10, 14, 6),
                            child: Row(
                              children: [
                                const Icon(
                                  Icons.tune_rounded,
                                  size: 14,
                                  color: Color(0xFF16DB65),
                                ),
                                const SizedBox(width: 6),
                                Text(
                                  'Mode Embedder',
                                  style: GoogleFonts.poppins(
                                    fontSize: 11.5,
                                    fontWeight: FontWeight.w600,
                                    color: const Color(0xFF16DB65),
                                  ),
                                ),
                              ],
                            ),
                          ),
                          const Divider(height: 1, color: Color(0xFF1E1E1E)),
                          // Improved option
                          InkWell(
                            onTap: _isUploading
                                ? null
                                : () => setDialogState(
                                      () => _embedderType = 'improved',
                                    ),
                            child: Padding(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 14,
                                vertical: 10,
                              ),
                              child: Row(
                                children: [
                                  AnimatedContainer(
                                    duration: const Duration(milliseconds: 150),
                                    width: 16,
                                    height: 16,
                                    decoration: BoxDecoration(
                                      shape: BoxShape.circle,
                                      border: Border.all(
                                        color: _embedderType == 'improved'
                                            ? const Color(0xFF16DB65)
                                            : const Color(0xFF444444),
                                        width: 2,
                                      ),
                                      color: _embedderType == 'improved'
                                          ? const Color(0xFF16DB65)
                                          : Colors.transparent,
                                    ),
                                    child: _embedderType == 'improved'
                                        ? const Icon(
                                            Icons.check,
                                            size: 10,
                                            color: Colors.black,
                                          )
                                        : null,
                                  ),
                                  const SizedBox(width: 10),
                                  Expanded(
                                    child: Column(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        Text(
                                          'Improved Embedder',
                                          style: GoogleFonts.poppins(
                                            fontSize: 12,
                                            fontWeight: FontWeight.w600,
                                            color: _embedderType == 'improved'
                                                ? Colors.white
                                                : const Color(0xFF888888),
                                          ),
                                        ),
                                        Text(
                                          'Akurasi lebih tinggi',
                                          style: GoogleFonts.poppins(
                                            fontSize: 10,
                                            color: const Color(0xFF555555),
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                          const Divider(height: 1, color: Color(0xFF1A1A1A)),
                          // Raw option
                          InkWell(
                            onTap: _isUploading
                                ? null
                                : () => setDialogState(
                                      () => _embedderType = 'raw',
                                    ),
                            child: Padding(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 14,
                                vertical: 10,
                              ),
                              child: Row(
                                children: [
                                  AnimatedContainer(
                                    duration: const Duration(milliseconds: 150),
                                    width: 16,
                                    height: 16,
                                    decoration: BoxDecoration(
                                      shape: BoxShape.circle,
                                      border: Border.all(
                                        color: _embedderType == 'raw'
                                            ? const Color(0xFFFFAA33)
                                            : const Color(0xFF444444),
                                        width: 2,
                                      ),
                                      color: _embedderType == 'raw'
                                          ? const Color(0xFFFFAA33)
                                          : Colors.transparent,
                                    ),
                                    child: _embedderType == 'raw'
                                        ? const Icon(
                                            Icons.check,
                                            size: 10,
                                            color: Colors.black,
                                          )
                                        : null,
                                  ),
                                  const SizedBox(width: 10),
                                  Expanded(
                                    child: Column(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        Text(
                                          'Raw Embedder',
                                          style: GoogleFonts.poppins(
                                            fontSize: 12,
                                            fontWeight: FontWeight.w600,
                                            color: _embedderType == 'raw'
                                                ? Colors.white
                                                : const Color(0xFF888888),
                                          ),
                                        ),
                                        Text(
                                          'Embed teks mentah langsung',
                                          style: GoogleFonts.poppins(
                                            fontSize: 10,
                                            color: const Color(0xFF555555),
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 12),
                    GestureDetector(
                      onTap: _isUploading ? null : pickFiles,
                      child: Container(
                        width: double.infinity,
                        padding: const EdgeInsets.symmetric(
                          vertical: 20,
                          horizontal: 16,
                        ),
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
                          ],
                        ),
                      ),
                    ),
                    if (hasFiles) ...[
                      const SizedBox(height: 12),
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
                      ConstrainedBox(
                        constraints: const BoxConstraints(maxHeight: 200),
                        child: ListView.separated(
                          shrinkWrap: true,
                          itemCount: _selectedFiles.length,
                          separatorBuilder: (_, __) => const SizedBox(height: 6),
                          itemBuilder: (_, i) {
                            final f = _selectedFiles[i];
                            final isDone = _isUploading && i < _uploadDone;
                            return Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 12,
                                vertical: 8,
                              ),
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
                    Row(
                      children: [
                        Expanded(
                          child: TextButton(
                            onPressed: _isUploading ? null : () => Navigator.of(ctx).pop(),
                            style: TextButton.styleFrom(
                              padding: const EdgeInsets.symmetric(vertical: 12),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(10),
                              ),
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
                            onPressed: (hasFiles && !_isUploading) ? doUpload : null,
                            style: ElevatedButton.styleFrom(
                              backgroundColor: (hasFiles && !_isUploading)
                                  ? const Color(0xFF16DB65)
                                  : const Color(0xFF1A1A1A),
                              padding: const EdgeInsets.symmetric(vertical: 12),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(10),
                              ),
                              elevation: 0,
                            ),
                            child: _isUploading
                                ? const SizedBox(
                                    width: 16,
                                    height: 16,
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
    final isMobile = MediaQuery.of(context).size.width < 768;
    final hasPending = widget.pendingDetailId != null;

    return Container(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 20),
      decoration: const BoxDecoration(
        color: Color(0xFF0D0D0D),
        border: Border(top: BorderSide(color: Color(0xFF1A1A1A))),
      ),
      child: isMobile
          ? _MobileInputLayout(
              hasText: _hasText,
              sending: widget.sending,
              hasPending: hasPending,
              isListening: _isListening,
              speechEnabled: _speechEnabled,
              modelSwitching: _modelSwitching,
              ragModeSwitching: _ragModeSwitching,
              ragMode: _ragMode,
              activeModelId: _activeModelId,
              selectedImageBytes: _selectedImageBytes, // Pass Image Bytes
              controller: widget.controller,
              focusNode: widget.focusNode,
              onSend: _triggerSend,
              onStop: widget.onStop,
              onStartListening: _startListening,
              onStopListening: _stopListening,
              onShowModelDialog: _showModelDialog,
              onShowUploadDialog: _showUploadDialog,
              onShowRagModeDialog: _showRagModeDialog,
              onPickImage: _pickImage, // Pass Image Picker action
              onClearImage: () => setState(() {
                _selectedImageBytes = null;
                _selectedImageName = null;
              }),
            )
          : _DesktopInputLayout(
              hasText: _hasText,
              sending: widget.sending,
              hasPending: hasPending,
              isListening: _isListening,
              speechEnabled: _speechEnabled,
              modelSwitching: _modelSwitching,
              ragModeSwitching: _ragModeSwitching,
              ragMode: _ragMode,
              activeModelId: _activeModelId,
              selectedImageBytes: _selectedImageBytes, // Pass Image Bytes
              controller: widget.controller,
              focusNode: widget.focusNode,
              onSend: _triggerSend,
              onStop: widget.onStop,
              onStartListening: _startListening,
              onStopListening: _stopListening,
              onShowModelDialog: _showModelDialog,
              onShowUploadDialog: _showUploadDialog,
              onShowRagModeDialog: _showRagModeDialog,
              onPickImage: _pickImage, // Pass Image Picker action
              onClearImage: () => setState(() {
                _selectedImageBytes = null;
                _selectedImageName = null;
              }),
            ),
    );
  }
}

// ---------------------------------------------------------------------------
// Mobile Input Layout
// ---------------------------------------------------------------------------

class _MobileInputLayout extends StatelessWidget {
  const _MobileInputLayout({
    required this.hasText,
    required this.sending,
    required this.hasPending,
    required this.isListening,
    required this.speechEnabled,
    required this.modelSwitching,
    required this.ragModeSwitching,
    required this.ragMode,
    required this.activeModelId,
    this.selectedImageBytes,
    required this.controller,
    required this.focusNode,
    required this.onSend,
    required this.onStop,
    required this.onStartListening,
    required this.onStopListening,
    required this.onShowModelDialog,
    required this.onShowUploadDialog,
    required this.onShowRagModeDialog,
    required this.onPickImage,
    required this.onClearImage,
  });

  final bool hasText;
  final bool sending;
  final bool hasPending;
  final bool isListening;
  final bool speechEnabled;
  final bool modelSwitching;
  final bool ragModeSwitching;
  final String ragMode;
  final String activeModelId;
  final Uint8List? selectedImageBytes;
  final TextEditingController controller;
  final FocusNode focusNode;
  final VoidCallback onSend;
  final VoidCallback? onStop;
  final VoidCallback onStartListening;
  final VoidCallback onStopListening;
  final Future<void> Function(BuildContext) onShowModelDialog;
  final void Function(BuildContext) onShowUploadDialog;
  final void Function(BuildContext) onShowRagModeDialog;
  final VoidCallback onPickImage;
  final VoidCallback onClearImage;

  String _getModelShortName(String modelId) {
    if (modelId.contains('mistral')) return 'Mistral';
    if (modelId.contains('qwen')) return 'Qwen';
    return 'Llama';
  }

  String _getRagModeShortName(String mode) {
    return mode == 'improved' ? 'Improved' : 'Regular';
  }

  @override
  Widget build(BuildContext context) {
    final isSendDisabled = sending || (!hasText && selectedImageBytes == null) || modelSwitching;

    return Column(
      children: [
        if (selectedImageBytes != null) ...[
          Container(
            alignment: Alignment.centerLeft,
            margin: const EdgeInsets.only(bottom: 12),
            child: Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: const Color(0xFF2A2A2A)),
              ),
              child: Stack(
                children: [
                  ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: Image.memory(
                      selectedImageBytes!,
                      height: 100,
                      width: 100,
                      fit: BoxFit.cover,
                    ),
                  ),
                  Positioned(
                    top: 4,
                    right: 4,
                    child: GestureDetector(
                      onTap: onClearImage,
                      child: Container(
                        padding: const EdgeInsets.all(4),
                        decoration: const BoxDecoration(
                          color: Colors.black87,
                          shape: BoxShape.circle,
                        ),
                        child: const Icon(Icons.close, size: 14, color: Colors.white),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
        Row(
          children: [
            Expanded(
              child: Tooltip(
                message: ragMode == 'improved' 
                    ? 'Mode: Improved' 
                    : 'Mode: Regular',
                child: ElevatedButton(
                  onPressed: ragModeSwitching ? null : () => onShowRagModeDialog(context),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF1A1A1A),
                    padding: const EdgeInsets.symmetric(vertical: 10),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    elevation: 0,
                  ),
                  child: ragModeSwitching
                      ? const SizedBox(
                          width: 14,
                          height: 14,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Color(0xFFFFAA33),
                          ),
                        )
                      : Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(
                              ragMode == 'improved' 
                                  ? Icons.auto_awesome_rounded 
                                  : Icons.speed_rounded,
                              size: 15,
                              color: ragMode == 'improved' 
                                  ? const Color(0xFF16DB65) 
                                  : const Color(0xFFFFAA33),
                            ),
                            const SizedBox(width: 5),
                            Flexible(
                              child: Text(
                                _getRagModeShortName(ragMode),
                                style: GoogleFonts.poppins(
                                  fontSize: 11,
                                  fontWeight: FontWeight.w500,
                                  color: ragMode == 'improved' 
                                      ? const Color(0xFF16DB65) 
                                      : const Color(0xFFFFAA33),
                                ),
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                          ],
                        ),
                ),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: Tooltip(
                message: 'Model: $activeModelId',
                child: ElevatedButton(
                  onPressed: modelSwitching ? null : () => onShowModelDialog(context),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF1A1A1A),
                    padding: const EdgeInsets.symmetric(vertical: 10),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    elevation: 0,
                  ),
                  child: modelSwitching
                      ? const SizedBox(
                          width: 14,
                          height: 14,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Color(0xFF16DB65),
                          ),
                        )
                      : Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            const Icon(
                              Icons.auto_awesome_rounded,
                              size: 15,
                              color: Color(0xFF16DB65),
                            ),
                            const SizedBox(width: 5),
                            Flexible(
                              child: Text(
                                _getModelShortName(activeModelId),
                                style: GoogleFonts.poppins(
                                  fontSize: 12,
                                  fontWeight: FontWeight.w500,
                                  color: const Color(0xFF16DB65),
                                ),
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                          ],
                        ),
                ),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: Tooltip(
                message: 'Pilih Gambar (Vision RAG)',
                child: ElevatedButton(
                  onPressed: sending ? null : onPickImage,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF1A1A1A),
                    padding: const EdgeInsets.symmetric(vertical: 10),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    elevation: 0,
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(
                        Icons.image_outlined,
                        color: Color(0xFF888888),
                        size: 18,
                      ),
                      const SizedBox(width: 5),
                      Text(
                        'Foto',
                        style: GoogleFonts.poppins(
                          fontSize: 11,
                          color: const Color(0xFF888888),
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 10),
        ConstrainedBox(
          constraints: const BoxConstraints(maxHeight: 75, minHeight: 70),
          child: TextField(
            controller: controller,
            focusNode: focusNode,
            maxLines: null,
            keyboardType: TextInputType.multiline,
            textInputAction: TextInputAction.send,
            enabled: !sending && !modelSwitching,
            style: GoogleFonts.poppins(
              fontSize: 15,
              color: Colors.white,
              height: 1.5,
            ),
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
                vertical: 16,
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
            onSubmitted: (_) {
              if (!isSendDisabled) {
                onSend();
              }
            },
          ),
        ),
        const SizedBox(height: 10),
        Row(
          children: [
            Expanded(
              child: Tooltip(
                message: 'Unggah PDF pengetahuan bot',
                child: ElevatedButton(
                  onPressed: sending ? null : () => onShowUploadDialog(context),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF1A1A1A),
                    padding: const EdgeInsets.symmetric(vertical: 10),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    elevation: 0,
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(
                        Icons.upload_file_rounded,
                        color: Color(0xFF888888),
                        size: 18,
                      ),
                      const SizedBox(width: 5),
                      Text(
                        'Upload PDF',
                        style: GoogleFonts.poppins(
                          fontSize: 11,
                          color: const Color(0xFF888888),
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: Tooltip(
                message: isListening ? 'Berhenti Merekam' : 'Input Suara',
                child: ElevatedButton(
                  onPressed: speechEnabled
                      ? (isListening ? onStopListening : onStartListening)
                      : null,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: isListening
                        ? Colors.red.shade400
                        : const Color(0xFF1A1A1A),
                    padding: const EdgeInsets.symmetric(vertical: 10),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    elevation: 0,
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        isListening ? Icons.mic_off_rounded : Icons.mic_none_rounded,
                        color: isListening ? Colors.white : const Color(0xFF888888),
                        size: 18,
                      ),
                      const SizedBox(width: 5),
                      Text(
                        isListening ? 'Stop' : 'Suara',
                        style: GoogleFonts.poppins(
                          fontSize: 11,
                          color: isListening ? Colors.white : const Color(0xFF888888),
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: hasPending
                  ? Tooltip(
                      message: 'Hentikan generate',
                      child: ElevatedButton(
                        onPressed: onStop,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFFFF4444),
                          padding: const EdgeInsets.symmetric(vertical: 10),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                          elevation: 0,
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            const Icon(
                              Icons.stop_rounded,
                              color: Colors.white,
                              size: 18,
                            ),
                            const SizedBox(width: 5),
                            Text(
                              'Stop',
                              style: GoogleFonts.poppins(
                                fontSize: 11,
                                color: Colors.white,
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          ],
                        ),
                      ),
                    )
                  : ElevatedButton(
                      onPressed: isSendDisabled ? null : onSend,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: (hasText || selectedImageBytes != null) && !sending && !modelSwitching
                            ? const Color(0xFF16DB65)
                            : const Color(0xFF1A1A1A),
                        padding: const EdgeInsets.symmetric(vertical: 10),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                        elevation: 0,
                      ),
                      child: (sending || modelSwitching)
                          ? const SizedBox(
                              width: 16,
                              height: 16,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                color: Colors.black,
                              ),
                            )
                          : Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                const Icon(
                                  Icons.arrow_upward_rounded,
                                  color: Colors.black,
                                  size: 18,
                                ),
                                const SizedBox(width: 5),
                                Text(
                                  'Kirim',
                                  style: GoogleFonts.poppins(
                                    fontSize: 11,
                                    fontWeight: FontWeight.w600,
                                    color: Colors.black,
                                  ),
                                ),
                              ],
                            ),
                    ),
            ),
          ],
        ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Desktop Input Layout
// ---------------------------------------------------------------------------

class _DesktopInputLayout extends StatelessWidget {
  const _DesktopInputLayout({
    required this.hasText,
    required this.sending,
    required this.hasPending,
    required this.isListening,
    required this.speechEnabled,
    required this.modelSwitching,
    required this.ragModeSwitching,
    required this.ragMode,
    required this.activeModelId,
    this.selectedImageBytes,
    required this.controller,
    required this.focusNode,
    required this.onSend,
    required this.onStop,
    required this.onStartListening,
    required this.onStopListening,
    required this.onShowModelDialog,
    required this.onShowUploadDialog,
    required this.onShowRagModeDialog,
    required this.onPickImage,
    required this.onClearImage,
  });

  final bool hasText;
  final bool sending;
  final bool hasPending;
  final bool isListening;
  final bool speechEnabled;
  final bool modelSwitching;
  final bool ragModeSwitching;
  final String ragMode;
  final String activeModelId;
  final Uint8List? selectedImageBytes;
  final TextEditingController controller;
  final FocusNode focusNode;
  final VoidCallback onSend;
  final VoidCallback? onStop;
  final VoidCallback onStartListening;
  final VoidCallback onStopListening;
  final Future<void> Function(BuildContext) onShowModelDialog;
  final void Function(BuildContext) onShowUploadDialog;
  final void Function(BuildContext) onShowRagModeDialog;
  final VoidCallback onPickImage;
  final VoidCallback onClearImage;

  String _getRagModeShortName(String mode) {
    return mode == 'improved' ? 'Improved' : 'Regular';
  }

  String _getModelShortName(String modelId) {
    if (modelId.contains('mistral')) return 'Mistral';
    if (modelId.contains('qwen')) return 'Qwen';
    return 'Llama';
  }

  @override
  Widget build(BuildContext context) {
    final isSendDisabled = sending || (!hasText && selectedImageBytes == null) || modelSwitching;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (selectedImageBytes != null) ...[
          Container(
            margin: const EdgeInsets.only(bottom: 12),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: const Color(0xFF2A2A2A)),
            ),
            child: Stack(
              children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: Image.memory(
                    selectedImageBytes!,
                    height: 100,
                    width: 100,
                    fit: BoxFit.cover,
                  ),
                ),
                Positioned(
                  top: 4,
                  right: 4,
                  child: GestureDetector(
                    onTap: onClearImage,
                    child: Container(
                      padding: const EdgeInsets.all(4),
                      decoration: const BoxDecoration(
                        color: Colors.black87,
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(Icons.close, size: 14, color: Colors.white),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
        Row(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Padding(
              padding: const EdgeInsets.only(bottom: 0),
              child: SizedBox(
                width: 48,
                height: 48,
                child: Tooltip(
                  message: 'Unggah PDF pengetahuan bot',
                  child: ElevatedButton(
                    onPressed: () => onShowUploadDialog(context),
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

            SizedBox(
              height: 48,
              child: Tooltip(
                message: ragMode == 'improved' 
                    ? 'Mode: Improved (Neo4j + Context)' 
                    : 'Mode: Regular (Simple)',
                child: ElevatedButton(
                  onPressed: ragModeSwitching ? null : () => onShowRagModeDialog(context),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF1A1A1A),
                    padding: const EdgeInsets.symmetric(horizontal: 10),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    elevation: 0,
                  ),
                  child: ragModeSwitching
                      ? const SizedBox(
                          width: 14,
                          height: 14,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Color(0xFFFFAA33),
                          ),
                        )
                      : Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(
                              ragMode == 'improved' 
                                  ? Icons.auto_awesome_rounded 
                                  : Icons.speed_rounded,
                              size: 15,
                              color: ragMode == 'improved' 
                                  ? const Color(0xFF16DB65) 
                                  : const Color(0xFFFFAA33),
                            ),
                            const SizedBox(width: 5),
                            Text(
                              _getRagModeShortName(ragMode),
                              style: GoogleFonts.poppins(
                                fontSize: 11,
                                fontWeight: FontWeight.w500,
                                color: ragMode == 'improved' 
                                    ? const Color(0xFF16DB65) 
                                    : const Color(0xFFFFAA33),
                              ),
                            ),
                          ],
                        ),
                ),
              ),
            ),
            const SizedBox(width: 8),

            SizedBox(
              height: 48,
              child: Tooltip(
                message: 'Model: $activeModelId',
                child: ElevatedButton(
                  onPressed: modelSwitching ? null : () => onShowModelDialog(context),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF1A1A1A),
                    padding: const EdgeInsets.symmetric(horizontal: 10),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    elevation: 0,
                  ),
                  child: modelSwitching
                      ? const SizedBox(
                          width: 14,
                          height: 14,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Color(0xFF16DB65),
                          ),
                        )
                      : Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            const Icon(
                              Icons.auto_awesome_rounded,
                              size: 15,
                              color: Color(0xFF16DB65),
                            ),
                            const SizedBox(width: 5),
                            Text(
                              _getModelShortName(activeModelId),
                              style: GoogleFonts.poppins(
                                fontSize: 12,
                                fontWeight: FontWeight.w500,
                                color: const Color(0xFF16DB65),
                              ),
                            ),
                          ],
                        ),
                ),
              ),
            ),
            const SizedBox(width: 8),
            
            Padding(
              padding: const EdgeInsets.only(bottom: 0),
              child: SizedBox(
                width: 48,
                height: 48,
                child: Tooltip(
                  message: 'Pilih Gambar (Vision RAG)',
                  child: ElevatedButton(
                    onPressed: sending ? null : onPickImage,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF1A1A1A),
                      padding: EdgeInsets.zero,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                      elevation: 0,
                    ),
                    child: const Icon(
                      Icons.image_outlined,
                      color: Color(0xFF888888),
                      size: 20,
                    ),
                  ),
                ),
              ),
            ),
            const SizedBox(width: 10),

            Expanded(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxHeight: 150),
                child: Shortcuts(
                  shortcuts: <LogicalKeySet, Intent>{
                    LogicalKeySet(LogicalKeyboardKey.enter): const SendMessageIntent(),
                    LogicalKeySet(LogicalKeyboardKey.numpadEnter): const SendMessageIntent(),
                  },
                  child: Actions(
                    actions: <Type, Action<Intent>>{
                      SendMessageIntent: CallbackAction<SendMessageIntent>(
                        onInvoke: (intent) {
                          if (!isSendDisabled) {
                            onSend();
                          }
                          return null;
                        },
                      ),
                    },
                    child: TextField(
                      controller: controller,
                      focusNode: focusNode,
                      maxLines: null,
                      keyboardType: TextInputType.multiline,
                      textInputAction: TextInputAction.send,
                      enabled: !sending && !modelSwitching,
                      style: GoogleFonts.poppins(
                        fontSize: 14,
                        color: Colors.white,
                      ),
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
                      onSubmitted: (_) {
                        if (!isSendDisabled) {
                          onSend();
                        }
                      },
                    ),
                  ),
                ),
              ),
            ),
            const SizedBox(width: 10),

            Padding(
              padding: const EdgeInsets.only(bottom: 0),
              child: SizedBox(
                width: 48,
                height: 48,
                child: Tooltip(
                  message: isListening ? 'Berhenti Merekam' : 'Input Suara',
                  child: ElevatedButton(
                    onPressed: speechEnabled
                        ? (isListening ? onStopListening : onStartListening)
                        : null,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: isListening
                          ? Colors.red.shade400
                          : const Color(0xFF1A1A1A),
                      padding: EdgeInsets.zero,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                      elevation: 0,
                    ),
                    child: Icon(
                      isListening ? Icons.mic_off_rounded : Icons.mic_none_rounded,
                      color: isListening ? Colors.white : const Color(0xFF888888),
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
                        onPressed: onStop,
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
                      onPressed: isSendDisabled ? null : onSend,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: (hasText || selectedImageBytes != null) && !sending && !modelSwitching
                            ? const Color(0xFF16DB65)
                            : const Color(0xFF1A1A1A),
                        padding: EdgeInsets.zero,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                        elevation: 0,
                      ),
                      child: (sending || modelSwitching)
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
      ],
    );
  }
}