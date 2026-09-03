/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
      colors: {
        background: '#f8f9ff',
        surface: {
          lowest: '#ffffff',
          low: '#eff4ff',
          DEFAULT: '#e5eeff',
          high: '#dce9ff',
        },
        'on-surface': '#0b1c30',
        'on-surface-variant': '#434655',
        outline: {
          DEFAULT: '#737687',
          variant: '#c3c6d8',
        },
        primary: {
          DEFAULT: '#0047c1',
          container: '#155eef',
        },
        'on-primary': '#ffffff',
        secondary: '#0051d5',
        error: {
          DEFAULT: '#ba1a1a',
          container: '#ffdad6',
        },
        'on-error-container': '#93000a',
        warning: {
          bg: '#fef3c7',
          text: '#92400e',
        },
        stable: {
          bg: '#dcfce7',
          text: '#166534',
        }
      },
      borderRadius: {
        xl: '24px',
        lg: '16px',
        full: '9999px',
      },
      boxShadow: {
        soft: '0px 4px 12px rgba(15,23,42,0.08)',
      },
    },
  },
  plugins: [],
}
