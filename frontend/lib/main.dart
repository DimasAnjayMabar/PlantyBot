import 'package:flutter/material.dart';
import 'package:frontend/chats/chats.dart';
import 'package:frontend/chats/user_profile.dart';
import 'package:frontend/users/change_email/change_email.dart';
import 'package:frontend/users/change_email/verify_otp.dart';
import 'package:frontend/users/forgot_password/request_otp.dart';
import 'package:frontend/users/forgot_password/reset_password.dart';
import 'package:frontend/users/forgot_password/verify_otp.dart';
import 'package:frontend/users/login/login.dart';
import 'package:frontend/users/register/register.dart';
import 'package:frontend/users/register/verify_otp.dart';
import 'package:go_router/go_router.dart';

void main() {
  runApp(const MyApp());
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

final GoRouter _router = GoRouter(
  initialLocation: '/users/login',
  routes: [
    // Auth – Register
    GoRoute(
      path: '/users/register',
      builder: (context, state) => const RegisterPage(),
      routes: [
        GoRoute(
          path: 'verify-otp', // resolves to /register/verify-otp
          builder: (context, state) {
            final email = state.uri.queryParameters['email'] ?? '';
            return RegisterVerifyOtpPage(email: email);
          },
        ),
      ],
    ),

    // Auth – Login
    GoRoute(
      path: '/users/login',
      builder: (context, state) => const LoginPage(),
    ),

    // Auth – Forgot Password
    GoRoute(
      path: '/users/reset-password/otp',
      builder: (context, state) => const ForgotPasswordPage(),
      routes: [
        GoRoute(
          path: 'verify-otp', // resolves to /forgot-password/verify-otp
          builder: (context, state) {
            final email = state.uri.queryParameters['email'] ?? '';
            return ForgotPasswordVerifyOtpPage(email: email);
          },
        ),
      ],
    ),

    // Auth – Reset Password
    GoRoute(
      path: '/users/reset-password',
      builder: (context, state) {
        final token = state.uri.queryParameters['token'] ?? '';
        return ResetPasswordPage(token: token);
      },
    ),

    // Change Email
    GoRoute(
      path: '/users/change-email/otp/verify-otp',
      builder: (context, state) {
        final email = state.uri.queryParameters['email'] ?? '';
        return ChangeEmailVerifyOtpPage(email: email);
      },
    ),
    
    GoRoute(
      path: '/users/change-email',
      builder: (context, state) {
        final token = state.uri.queryParameters['token'] ?? '';
        return ChangeEmailPage(token: token);
      },
    ),

    // Main
    GoRoute(path: '/chats', builder: (context, state) => const ChatsPage()),

    GoRoute(path: '/user_profile', builder: (context, state) => const UserProfilePage()),
  ],
);

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp.router(
      title: 'My App',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
      ),
      routerConfig: _router,
    );
  }
}
