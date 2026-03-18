import { LoginForm } from '@/components/auth/LoginForm'

export default function LoginPage() {
  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'var(--bg)',
      padding: '24px',
      position: 'relative',
      overflow: 'hidden',
    }}>
      {/* Ambient glow */}
      <div style={{
        position: 'absolute',
        top: '30%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        width: '600px',
        height: '600px',
        background: 'radial-gradient(ellipse at center, rgba(245,166,35,0.07) 0%, rgba(28,32,100,0.12) 40%, transparent 70%)',
        pointerEvents: 'none',
        borderRadius: '50%',
        filter: 'blur(40px)',
      }} />

      <LoginForm />
    </div>
  )
}
