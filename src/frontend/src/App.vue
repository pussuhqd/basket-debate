<template>
  <div id="app">
    <!-- HEADER -->
    <header class="header">
      <h1>🛒 Мультиагентный Шоппер</h1>
      <p>ИИ за 2 секунды найдёт лучшую корзину</p>
    </header>

    <!-- MAIN -->
    <div class="container">
      <!-- ЛЕВАЯ ПАНЕЛЬ -->
      <aside class="sidebar">
        <h2>Ваш запрос</h2>

        <div class="form-group">
          <label>Что вам нужно?</label>
          <input 
            v-model="userQuery"
            placeholder="ужин на двоих за 1500 руб"
            @keyup.enter="optimizeBasket"
            class="input"
          />
        </div>

        <button @click="optimizeBasket" :disabled="loading" class="btn-primary">
          {{ loading ? '⏳ Думаю...' : '🚀 Оптимизировать' }}
        </button>

        <hr class="divider">

        <h3>Параметры</h3>

        <div class="form-group">
          <label>Диета:</label>
          <select v-model="diet" class="input">
            <option>любая</option>
            <option>веган</option>
            <option>вегетарианец</option>
            <option>кето</option>
          </select>
        </div>

        <div class="form-group">
          <label>Аллергии:</label>
          <input 
            v-model="allergies"
            placeholder="молоко, орехи"
            class="input"
          />
        </div>
      </aside>

      <!-- ПРАВАЯ ПАНЕЛЬ -->
      <main class="content">
        <!-- Loading -->
        <div v-if="loading" class="state-loading">
          <div class="spinner"></div>
          <p class="loading-text">🤖 Три агента обсуждают вашу корзину...</p>
          <p class="loading-desc">
            💰 Budget Agent ищет дешевле<br>
            🔗 Compatibility Agent проверяет совместимость<br>
            👤 Profile Agent учитывает ваши предпочтения
          </p>
        </div>

        <!-- Error -->
        <div v-else-if="error" class="state-error">
          <p class="error-text">{{ error }}</p>
        </div>

        <!-- Success -->
        <div v-else-if="basket.length > 0" class="state-success">
          <h2>✅ Оптимальная корзина</h2>

          <div class="products">
            <div 
              v-for="item in basket"
              :key="item.id"
              class="product-card"
            >
              <div class="product-top">
                <h3>{{ item.name }}</h3>
                <span class="badge" :class="'badge-' + item.agent">
                  {{ agentLabel[item.agent] }}
                </span>
              </div>
              <p class="product-reason">{{ item.reason }}</p>
              <div class="product-bottom">
                <span class="price">{{ formatPrice(item.price) }} ₽</span>
                <span class="rating">⭐ {{ item.rating || 4.5 }}</span>
              </div>
            </div>
          </div>

          <div class="summary">
            <div class="summary-row">
              <span>Товаров:</span>
              <strong>{{ basket.length }}</strong>
            </div>
            <div class="summary-row">
              <span>Сумма:</span>
              <strong class="price">{{ formatPrice(totalPrice) }} ₽</strong>
            </div>
            <div class="summary-row">
              <span>Экономия:</span>
              <strong class="savings">-{{ formatPrice(originalPrice - totalPrice) }} ₽</strong>
            </div>
          </div>

          <button @click="addToCart" class="btn-secondary">
            🛍️ Добавить в корзину
          </button>
        </div>

        <!-- Empty -->
        <div v-else class="state-empty">
          <p class="empty-text">📋 Введите запрос и нажмите кнопку</p>
        </div>
      </main>
    </div>
  </div>
</template>

<script setup>
import { useBasket } from './composables/useBasket'
import './App.css'

// Импортируем всё из composable
const {
  userQuery,
  basket,
  loading,
  error,
  diet,
  allergies,
  originalPrice,
  totalPrice,
  agentLabel,
  optimizeBasket,
  formatPrice,
  addToCart
} = useBasket()
</script>
