import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: {
    default: 'underdogged — Football predictions powered by data',
    template: '%s | underdogged',
  },
  description:
    'Data-driven match predictions and value bet finder for the Premier League, Championship, Bundesliga, Serie A, and La Liga.',
  metadataBase: new URL('https://underdogged.vercel.app'),
  openGraph: {
    title: 'underdogged — Football predictions powered by data',
    description:
      'Data-driven match predictions and value bet finder for the Premier League, Championship, Bundesliga, Serie A, and La Liga.',
    url: 'https://underdogged.vercel.app',
    siteName: 'underdogged',
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'underdogged — football predictions powered by data',
      },
    ],
    locale: 'en_GB',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'underdogged — Football predictions powered by data',
    description:
      'Data-driven match predictions and value bet finder for five European leagues.',
    images: ['/og-image.png'],
  },
  icons: {
    icon: [
      { url: '/favicon-16x16.png', sizes: '16x16', type: 'image/png' },
      { url: '/favicon-32x32.png', sizes: '32x32', type: 'image/png' },
    ],
    apple: [
      { url: '/apple-touch-icon.png', sizes: '180x180', type: 'image/png' },
    ],
    shortcut: '/favicon.ico',
  },
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        {/* Set theme before first paint to avoid flash */}
        <script dangerouslySetInnerHTML={{ __html: `
          (function() {
            try {
              var t = localStorage.getItem('theme');
              if (!t) t = window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
              if (t === 'light') document.documentElement.setAttribute('data-theme', 'light');
            } catch(e) {}
          })();
        `}} />
      </head>
      <body>{children}</body>
    </html>
  )
}
