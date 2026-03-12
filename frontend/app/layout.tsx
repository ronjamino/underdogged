import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'underdogged',
  description: 'Match predictions. Five leagues.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
