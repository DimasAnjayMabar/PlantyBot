import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

final _dio = Dio(
  BaseOptions(
    baseUrl: 'http://localhost:8000',
    connectTimeout: const Duration(seconds: 10),
    receiveTimeout: const Duration(seconds: 10),
    headers: {'Content-Type': 'application/json'},
  ),
);

// ---------------------------------------------------------------------------
// Warna & konstanta
// ---------------------------------------------------------------------------

const _bg        = Color(0xFF020202);
const _neon      = Color(0xFF16DB65);
const _neonDim   = Color(0x3316DB65);
const _surface   = Color(0xFF0D0D0D);
const _border    = Color(0xFF16DB65);
const _textMuted = Color(0xFFA3A3A3);

const _otpLength = 6;

// ---------------------------------------------------------------------------
// ForgotPasswordVerifyOtpPage
// ---------------------------------------------------------------------------

class ForgotPasswordVerifyOtpPage extends StatefulWidget {
  const ForgotPasswordVerifyOtpPage({super.key, required this.email});

  final String email;

  @override
  State<ForgotPasswordVerifyOtpPage> createState() =>
      _ForgotPasswordVerifyOtpPageState();
}

class _ForgotPasswordVerifyOtpPageState
    extends State<ForgotPasswordVerifyOtpPage>
    with SingleTickerProviderStateMixin {
  final _otpController = TextEditingController();
  final _otpFocusNode = FocusNode();

  bool _isVerifying   = false;
  bool _isResending   = false;
  bool _resendEnabled = true;

  late final AnimationController _fadeController;
  late final Animation<double>   _fadeAnimation;

  @override
  void initState() {
    super.initState();
    _fadeController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    )..forward();
    _fadeAnimation = CurvedAnimation(
      parent: _fadeController,
      curve: Curves.easeOut,
    );
  }

  @override
  void dispose() {
    _fadeController.dispose();
    _otpController.dispose();
    _otpFocusNode.dispose();
    super.dispose();
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  String get _otpValue    => _otpController.text.trim();
  bool   get _otpComplete => _otpValue.length == _otpLength;

  // ── API calls ─────────────────────────────────────────────────────────────

  Future<void> _handleVerify() async {
    if (!_otpComplete) return;
    setState(() => _isVerifying = true);
    try {
      final response = await _dio.post(
        '/users/reset-password/verify-otp',
        data: {'email': widget.email, 'otp': _otpValue},
      );
      if (response.statusCode == 200 && mounted) {
        final resetToken = response.data['data']['reset_token'] as String;
        final token = Uri.encodeComponent(resetToken);
        context.push('/users/reset-password?token=$token');
      }
    } on DioException catch (e) {
      if (!mounted) return;
      String message = 'Verifikasi gagal. Coba lagi.';
      if (e.type == DioExceptionType.connectionTimeout ||
          e.type == DioExceptionType.receiveTimeout) {
        message = 'Koneksi timeout. Periksa jaringan kamu.';
      } else if (e.response?.data['detail'] != null) {
        message = e.response!.data['detail'].toString();
      }
      _showErrorSnackbar(message);
      _otpController.clear();
      setState(() {});
    } catch (_) {
      if (mounted) _showErrorSnackbar('Terjadi kesalahan tidak terduga.');
    } finally {
      if (mounted) setState(() => _isVerifying = false);
    }
  }

  Future<void> _handleResend() async {
    if (!_resendEnabled || _isResending) return;
    setState(() => _isResending = true);
    try {
      await _dio.post(
        '/users/reset-password/request-otp',
        data: {'email': widget.email},
      );
      if (mounted) {
        _showSuccessSnackbar('OTP baru telah dikirim ke email kamu.');
      }
    } on DioException catch (e) {
      if (!mounted) return;
      String message = 'Gagal mengirim ulang OTP.';
      if (e.response?.data['detail'] != null) {
        message = e.response!.data['detail'].toString();
      }
      if (e.response?.statusCode == 429) {
        setState(() => _resendEnabled = false);
        Future.delayed(const Duration(seconds: 5), () {
          if (mounted) setState(() => _resendEnabled = true);
        });
      }
      _showErrorSnackbar(message);
    } catch (_) {
      if (mounted) _showErrorSnackbar('Terjadi kesalahan tidak terduga.');
    } finally {
      if (mounted) setState(() => _isResending = false);
    }
  }

  // ── Snackbars ─────────────────────────────────────────────────────────────

  void _showErrorSnackbar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          message,
          style: GoogleFonts.poppins(fontSize: 13, color: Colors.white),
        ),
        backgroundColor: const Color(0xFF1A1A1A),
        behavior: SnackBarBehavior.floating,
        margin: const EdgeInsets.all(16),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(10),
          side: const BorderSide(color: Color(0xFFFF4D4D), width: 1),
        ),
      ),
    );
  }

  void _showSuccessSnackbar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          message,
          style: GoogleFonts.poppins(fontSize: 13, color: Colors.black),
        ),
        backgroundColor: _neon,
        behavior: SnackBarBehavior.floating,
        margin: const EdgeInsets.all(16),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      ),
    );
  }

  // ── Keyboard handler ──────────────────────────────────────────────────────
  
  void _handleKeyEvent(KeyEvent event) {
    if (event is KeyDownEvent && 
        event.logicalKey == LogicalKeyboardKey.enter &&
        _otpComplete &&
        !_isVerifying) {
      _handleVerify();
    }
  }

  // ── Build ─────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return KeyboardListener(
      focusNode: FocusNode()..requestFocus(),
      onKeyEvent: _handleKeyEvent,
      child: Scaffold(
        backgroundColor: _bg,
        body: SafeArea(
          child: FadeTransition(
            opacity: _fadeAnimation,
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 40),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // ── Back button ──────────────────────────────────────────
                  GestureDetector(
                    onTap: () => context.pop(),
                    child: Container(
                      width: 40,
                      height: 40,
                      decoration: BoxDecoration(
                        color: _neonDim,
                        borderRadius: BorderRadius.circular(12),
                        boxShadow: [
                          BoxShadow(
                            color: _neon.withValues(alpha: 0.25),
                            blurRadius: 12,
                            spreadRadius: 0,
                          ),
                        ],
                      ),
                      child: const Icon(
                        Icons.arrow_back_rounded,
                        color: _neon,
                        size: 24,
                      ),
                    ),
                  ),
                  const SizedBox(height: 48),

                  // ── Title ─────────────────────────────────────────────────
                  Text(
                    'Verifikasi OTP',
                    style: GoogleFonts.poppins(
                      fontSize: 28,
                      fontWeight: FontWeight.w700,
                      color: Colors.white,
                      letterSpacing: -0.5,
                    ),
                  ),
                  const SizedBox(height: 8),
                  RichText(
                    text: TextSpan(
                      style: GoogleFonts.poppins(
                        fontSize: 13,
                        color: _textMuted,
                        height: 1.6,
                      ),
                      children: [
                        const TextSpan(
                          text: 'Kode OTP reset password telah dikirim ke\n',
                        ),
                        TextSpan(
                          text: widget.email,
                          style: GoogleFonts.poppins(
                            fontSize: 13,
                            color: _neon,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 32),

                  // ── Info card ────────────────────────────────────────────
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 14,
                    ),
                    decoration: BoxDecoration(
                      color: _neonDim,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                        color: _neon.withValues(alpha: 0.25),
                        width: 1,
                      ),
                    ),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Icon(
                          Icons.info_outline_rounded,
                          color: _neon,
                          size: 18,
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: Text(
                            'Kode OTP berlaku selama 10 menit dan hanya dapat '
                            'diminta 5 kali per hari.',
                            style: GoogleFonts.poppins(
                              fontSize: 12,
                              color: _neon,
                              height: 1.5,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 40),

                  // ── OTP Field ─────────────────────────────────────────────
                  _NeonField(
                    controller: _otpController,
                    focusNode: _otpFocusNode,
                    label: 'Kode OTP',
                    hint: 'Masukkan 6 digit kode OTP',
                    icon: Icons.pin_outlined,
                    keyboardType: TextInputType.number,
                    maxLength: _otpLength,
                    textInputAction: TextInputAction.done,
                    onFieldSubmitted: (_) => _handleVerify(),
                    onChanged: (_) => setState(() {}),
                  ),
                  const SizedBox(height: 40),

                  // ── Verify button ─────────────────────────────────────────
                  _NeonButton(
                    label: 'Verifikasi',
                    isLoading: _isVerifying,
                    enabled: _otpComplete,
                    onPressed: _handleVerify,
                  ),
                  const SizedBox(height: 28),

                  // ── Resend ────────────────────────────────────────────────
                  Center(
                    child: _isResending
                        ? SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: _neon.withValues(alpha: 0.6),
                            ),
                          )
                        : RichText(
                            text: TextSpan(
                              style: GoogleFonts.poppins(
                                fontSize: 13,
                                color: _textMuted,
                              ),
                              children: [
                                const TextSpan(text: 'Tidak menerima kode? '),
                                WidgetSpan(
                                  alignment: PlaceholderAlignment.middle,
                                  child: GestureDetector(
                                    onTap: _resendEnabled ? _handleResend : null,
                                    child: Text(
                                      'Kirim ulang',
                                      style: GoogleFonts.poppins(
                                        fontSize: 13,
                                        fontWeight: FontWeight.w600,
                                        color: _resendEnabled
                                            ? _neon
                                            : _textMuted,
                                      ),
                                    ),
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
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Neon TextField
// ---------------------------------------------------------------------------

class _NeonField extends StatefulWidget {
  const _NeonField({
    required this.controller,
    required this.label,
    required this.hint,
    required this.icon,
    this.focusNode,
    this.keyboardType,
    this.maxLength,
    this.onChanged,
    this.textInputAction,
    this.onFieldSubmitted,
  });

  final TextEditingController controller;
  final String label;
  final String hint;
  final IconData icon;
  final FocusNode? focusNode;
  final TextInputType? keyboardType;
  final int? maxLength;
  final ValueChanged<String>? onChanged;
  final TextInputAction? textInputAction;
  final void Function(String)? onFieldSubmitted;

  @override
  State<_NeonField> createState() => _NeonFieldState();
}

class _NeonFieldState extends State<_NeonField> {
  bool _focused = false;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          widget.label,
          style: GoogleFonts.poppins(
            fontSize: 12,
            fontWeight: FontWeight.w500,
            color: _focused ? _neon : _textMuted,
            letterSpacing: 0.3,
          ),
        ),
        const SizedBox(height: 8),
        AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(12),
            boxShadow: _focused
                ? [BoxShadow(color: _neon.withValues(alpha: 0.25), blurRadius: 16)]
                : [],
          ),
          child: Focus(
            onFocusChange: (f) => setState(() => _focused = f),
            child: TextFormField(
              controller: widget.controller,
              focusNode: widget.focusNode,
              keyboardType: widget.keyboardType,
              maxLength: widget.maxLength,
              onChanged: widget.onChanged,
              textInputAction: widget.textInputAction,
              onFieldSubmitted: widget.onFieldSubmitted,
              textAlign: TextAlign.center,
              style: GoogleFonts.poppins(
                fontSize: 22,
                fontWeight: FontWeight.w700,
                color: _neon,
                letterSpacing: 8,
              ),
              cursorColor: _neon,
              decoration: InputDecoration(
                hintText: widget.hint,
                hintStyle: GoogleFonts.poppins(
                  fontSize: 14,
                  color: _textMuted,
                  letterSpacing: 0,
                ),
                prefixIcon: Icon(
                  widget.icon,
                  color: _focused ? _neon : _textMuted,
                  size: 20,
                ),
                counterText: '',
                filled: true,
                fillColor: _surface,
                contentPadding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 20,
                ),
                enabledBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: const BorderSide(
                    color: Color(0xFF1A1A1A),
                    width: 1.5,
                  ),
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: const BorderSide(color: _border, width: 1.5),
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Neon Button
// ---------------------------------------------------------------------------

class _NeonButton extends StatelessWidget {
  const _NeonButton({
    required this.label,
    required this.onPressed,
    this.isLoading = false,
    this.enabled = true,
  });

  final String label;
  final VoidCallback onPressed;
  final bool isLoading;
  final bool enabled;

  @override
  Widget build(BuildContext context) {
    final bool active = enabled && !isLoading;

    return SizedBox(
      width: double.infinity,
      height: 52,
      child: DecoratedBox(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(14),
          boxShadow: active
              ? [
                  BoxShadow(
                    color: _neon.withValues(alpha: 0.35),
                    blurRadius: 20,
                    spreadRadius: 0,
                    offset: const Offset(0, 4),
                  ),
                ]
              : [],
        ),
        child: ElevatedButton(
          onPressed: active ? onPressed : null,
          style: ElevatedButton.styleFrom(
            backgroundColor: _neon,
            disabledBackgroundColor: _neon.withValues(alpha: 0.25),
            foregroundColor: Colors.black,
            elevation: 0,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(14),
            ),
          ),
          child: isLoading
              ? const SizedBox(
                  width: 22,
                  height: 22,
                  child: CircularProgressIndicator(
                    strokeWidth: 2.5,
                    color: Colors.black,
                  ),
                )
              : Text(
                  label,
                  style: GoogleFonts.poppins(
                    fontSize: 15,
                    fontWeight: FontWeight.w700,
                    color: active ? Colors.black : Colors.black45,
                    letterSpacing: 0.3,
                  ),
                ),
        ),
      ),
    );
  }
}