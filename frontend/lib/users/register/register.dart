// lib/users/register.dart
import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
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

const _bg = Color(0xFF020202);
const _neon = Color(0xFF16DB65);
const _neonDim = Color(0x3316DB65); // neon 20% opacity — untuk glow layer
const _surface = Color(0xFF0D0D0D);
const _border = Color(0xFF16DB65);
const _textMuted = Color(0xFFA3A3A3);

// ---------------------------------------------------------------------------
// RegisterPage
// ---------------------------------------------------------------------------

class RegisterPage extends StatefulWidget {
  const RegisterPage({super.key});

  @override
  State<RegisterPage> createState() => _RegisterPageState();
}

class _RegisterPageState extends State<RegisterPage>
    with SingleTickerProviderStateMixin {
  final _formKey = GlobalKey<FormState>();

  final _nameController = TextEditingController();
  final _usernameController = TextEditingController();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();

  bool _obscurePassword = true;
  bool _isLoading = false;

  late final AnimationController _fadeController;
  late final Animation<double> _fadeAnimation;

  @override
  void initState() {
    super.initState();
    _fadeController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    );
    _fadeAnimation = CurvedAnimation(
      parent: _fadeController,
      curve: Curves.easeOut,
    );
    _fadeController.forward();
  }

  @override
  void dispose() {
    _fadeController.dispose();
    _nameController.dispose();
    _usernameController.dispose();
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  void _handleRegister() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() => _isLoading = true);

    try {
      final response = await _dio.post(
        '/users/register',
        data: {
          'name': _nameController.text.trim(),
          'username': _usernameController.text.trim(),
          'email': _emailController.text.trim(),
          'password': _passwordController.text,
        },
      );

      if (response.statusCode == 201 && mounted) {
        final email = Uri.encodeComponent(_emailController.text.trim());
        context.push('/users/register/verify-otp?email=$email');
      }
    } on DioException catch (e) {
      if (!mounted) return;
      String message = 'Terjadi kesalahan. Coba lagi.';
      if (e.type == DioExceptionType.connectionTimeout ||
          e.type == DioExceptionType.receiveTimeout) {
        message = 'Koneksi timeout. Periksa jaringan kamu.';
      } else if (e.response?.data['detail'] != null) {
        message = e.response!.data['detail'].toString();
      }
      _showErrorSnackbar(message);
    } catch (_) {
      if (mounted) _showErrorSnackbar('Terjadi kesalahan tidak terduga.');
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _bg,
      body: SafeArea(
        child: FadeTransition(
          opacity: _fadeAnimation,
          child: SingleChildScrollView(
            padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 40),
            child: Form(
              key: _formKey,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // ── Back Button (replaces logo & title) ─────────────────
                  const _BackButton(),
                  const SizedBox(height: 40),

                  // ── Heading ─────────────────────────────────────────────
                  Text(
                    'Buat Akun',
                    style: GoogleFonts.poppins(
                      fontSize: 28,
                      fontWeight: FontWeight.w700,
                      color: Colors.white,
                      letterSpacing: -0.5,
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    'Daftarkan akun untuk mulai menggunakan Ragna.',
                    style: GoogleFonts.poppins(
                      fontSize: 13,
                      color: _textMuted,
                      height: 1.5,
                    ),
                  ),
                  const SizedBox(height: 36),

                  // ── Fields ───────────────────────────────────────────────
                  _NeonField(
                    controller: _nameController,
                    label: 'Nama Lengkap',
                    hint: 'John Doe',
                    icon: Icons.person_outline_rounded,
                    textInputAction: TextInputAction.next,
                    onFieldSubmitted: (_) => FocusScope.of(context).nextFocus(),
                    validator: (v) =>
                        (v == null || v.isEmpty) ? 'Nama tidak boleh kosong' : null,
                  ),
                  const SizedBox(height: 20),

                  _NeonField(
                    controller: _usernameController,
                    label: 'Username',
                    hint: '@johndoe',
                    icon: Icons.alternate_email_rounded,
                    textInputAction: TextInputAction.next,
                    onFieldSubmitted: (_) => FocusScope.of(context).nextFocus(),
                    validator: (v) => (v == null || v.isEmpty)
                        ? 'Username tidak boleh kosong'
                        : null,
                  ),
                  const SizedBox(height: 20),

                  _NeonField(
                    controller: _emailController,
                    label: 'Email',
                    hint: 'johndoe@email.com',
                    icon: Icons.mail_outline_rounded,
                    keyboardType: TextInputType.emailAddress,
                    textInputAction: TextInputAction.next,
                    onFieldSubmitted: (_) => FocusScope.of(context).nextFocus(),
                    validator: (v) {
                      if (v == null || v.isEmpty) return 'Email tidak boleh kosong';
                      if (!v.contains('@')) return 'Format email tidak valid';
                      return null;
                    },
                  ),
                  const SizedBox(height: 20),

                  _NeonField(
                    controller: _passwordController,
                    label: 'Password',
                    hint: '••••••••',
                    icon: Icons.lock_outline_rounded,
                    obscureText: _obscurePassword,
                    textInputAction: TextInputAction.done,
                    onFieldSubmitted: (_) => _handleRegister(), // Enter key triggers registration
                    suffixIcon: IconButton(
                      icon: Icon(
                        _obscurePassword
                            ? Icons.visibility_off_outlined
                            : Icons.visibility_outlined,
                        color: _neon,
                        size: 20,
                      ),
                      onPressed: () =>
                          setState(() => _obscurePassword = !_obscurePassword),
                    ),
                    validator: (v) {
                      if (v == null || v.isEmpty) return 'Password tidak boleh kosong';
                      if (v.length < 8) return 'Password minimal 8 karakter';
                      return null;
                    },
                  ),
                  const SizedBox(height: 36),

                  // ── Submit button ────────────────────────────────────────
                  _NeonButton(
                    label: 'Daftar',
                    isLoading: _isLoading,
                    onPressed: _handleRegister,
                  ),
                  const SizedBox(height: 28),

                  // ── Footer ───────────────────────────────────────────────
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text(
                        'Sudah punya akun? ',
                        style: GoogleFonts.poppins(
                          fontSize: 13,
                          color: _textMuted,
                        ),
                      ),
                      GestureDetector(
                        onTap: () {
                          context.go('/users/login');
                        },
                        child: Text(
                          'Masuk',
                          style: GoogleFonts.poppins(
                            fontSize: 13,
                            color: _neon,
                            fontWeight: FontWeight.w600,
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
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Back Button Widget (Replaces Logo & Title)
// ---------------------------------------------------------------------------

class _BackButton extends StatelessWidget {
  const _BackButton();

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () {
        // Navigate back to the previous screen (e.g., login or home)
        if (Navigator.canPop(context)) {
          Navigator.pop(context);
        } else {
          // Fallback: go to login page if no back history exists
          context.go('/users/login');
        }
      },
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Neon-styled back icon inside a subtle container
          Container(
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
          const SizedBox(width: 12),
        ],
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
    this.obscureText = false,
    this.keyboardType,
    this.suffixIcon,
    this.validator,
    this.textInputAction,
    this.onFieldSubmitted,
  });

  final TextEditingController controller;
  final String label;
  final String hint;
  final IconData icon;
  final bool obscureText;
  final TextInputType? keyboardType;
  final Widget? suffixIcon;
  final String? Function(String?)? validator;
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
        // Label
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

        // Field dengan glow
        AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(12),
            boxShadow: _focused
                ? [
                    BoxShadow(
                      color: _neon.withValues(alpha: 0.25),
                      blurRadius: 16,
                      spreadRadius: 0,
                    ),
                  ]
                : [],
          ),
          child: Focus(
            onFocusChange: (f) => setState(() => _focused = f),
            child: TextFormField(
              controller: widget.controller,
              obscureText: widget.obscureText,
              keyboardType: widget.keyboardType,
              textInputAction: widget.textInputAction,
              onFieldSubmitted: widget.onFieldSubmitted,
              validator: widget.validator,
              style: GoogleFonts.poppins(
                fontSize: 14,
                color: _neon,
                fontWeight: FontWeight.w500,
              ),
              cursorColor: _neon,
              decoration: InputDecoration(
                hintText: widget.hint,
                hintStyle: GoogleFonts.poppins(fontSize: 14, color: _textMuted),
                prefixIcon: Icon(
                  widget.icon,
                  color: _focused ? _neon : _textMuted,
                  size: 20,
                ),
                suffixIcon: widget.suffixIcon,
                filled: true,
                fillColor: _surface,
                contentPadding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 16,
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
                errorBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide(
                    color: Colors.red.shade700,
                    width: 1.5,
                  ),
                ),
                focusedErrorBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide(
                    color: Colors.red.shade700,
                    width: 1.5,
                  ),
                ),
                errorStyle: GoogleFonts.poppins(
                  fontSize: 11,
                  color: Colors.red.shade400,
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
  });

  final String label;
  final VoidCallback onPressed;
  final bool isLoading;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      height: 52,
      child: DecoratedBox(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(14),
          boxShadow: [
            BoxShadow(
              color: _neon.withValues(alpha: 0.35),
              blurRadius: 20,
              spreadRadius: 0,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: ElevatedButton(
          onPressed: isLoading ? null : onPressed,
          style: ElevatedButton.styleFrom(
            backgroundColor: _neon,
            disabledBackgroundColor: _neon.withValues(alpha: 0.5),
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
                    color: Colors.black,
                    letterSpacing: 0.3,
                  ),
                ),
        ),
      ),
    );
  }
}