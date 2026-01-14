import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      // Все /api/* → http://localhost:5000/
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,  // Меняем origin header
        secure: false,
        rewrite: (path) => path.replace(/^\/api/, '')  // /api/optimize → /optimize
      }
    }
  }
})
