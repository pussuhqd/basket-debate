<template>
  <div class="container">
    <h1>🛒 MAS Оптимизатор Корзины</h1>
    
    <form @submit.prevent="runSimulation">
      <div class="input-group">
        <label>Бюджет (₽):</label>
        <input v-model.number="form.budget" type="number" min="0" required />
      </div>
      
      <div class="input-group">
        <label>Шаги симуляции:</label>
        <input v-model.number="form.max_steps" type="number" min="1" max="20" />
      </div>
      
      <button type="submit" :disabled="loading">
        {{ loading ? '🤖 Агенты думают...' : '🚀 Оптимизировать' }}
      </button>
    </form>
    
    <!-- Результат -->
    <div v-if="result" class="result">
      <h2>✅ Корзина готова!</h2>
      
      <div class="metrics">
        <div class="metric">
          <span class="label">Сумма</span>
          <span class="value">{{ formatPrice(result.final_sum) }}</span>
        </div>
        <div class="metric">
          <span class="label">Товаров</span>
          <span class="value">{{ result.cart_size }}</span>
        </div>
      </div>
      
      <h3>📊 Оценки Агентов</h3>
      <div class="agents">
        <div class="agent budget" :title="'Штраф за отклонение от бюджета'">
          <strong>Бюджет</strong>
          <span>{{ result.total_rewards.budget_agent.toFixed(2) }}</span>
        </div>
        <div class="agent compat">
          <strong>Совместимость</strong>
          <span>{{ result.total_rewards.compat_agent.toFixed(2) }}</span>
        </div>
        <div class="agent profile">
          <strong>Профиль</strong>
          <span>{{ result.total_rewards.profile_agent.toFixed(2) }}</span>
        </div>
      </div>
    </div>
    
    <div v-if="error" class="error">
      ❌ {{ error }}
    </div>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'

// Reactive state
const form = reactive({
  budget: 1500,
  max_steps: 5
})

const loading = ref(false)
const result = ref(null)
const error = ref(null)

const API_BASE = 'http://localhost:5000'  // Backend URL

async function runSimulation() {
  loading.value = true
  error.value = null
  result.value = null
  
  console.log('🚀 Запуск симуляции:', form)  // Debug
  
  try {
    const response = await fetch(`${API_BASE}/optimize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(form)
    })
    
    console.log('📡 Response status:', response.status)  // Debug
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }
    
    const data = await response.json()  // ← Парсим JSON!
    console.log('✅ Получен результат:', data)  // Debug
    
    if (data.status === 'success') {
      result.value = data
    } else {
      error.value = data.message || 'Неизвестная ошибка'
    }
  } catch (err) {
    console.error('❌ Ошибка:', err)
    error.value = `Ошибка связи: ${err.message}`
  } finally {
    loading.value = false
  }
}

function formatPrice(price) {
  return new Intl.NumberFormat('ru-RU', {
    style: 'currency',
    currency: 'RUB'
  }).format(price)
}
</script>

<style scoped>
.container {
  max-width: 600px;
  margin: 0 auto;
  padding: 2rem;
  font-family: -apple-system, BlinkMacSystemFont, sans-serif;
}

h1 {
  text-align: center;
  color: #2c3e50;
  margin-bottom: 2rem;
}

form {
  background: white;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  margin-bottom: 2rem;
}

.input-group {
  margin-bottom: 1.5rem;
}

label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 500;
  color: #374151;
}

input {
  width: 100%;
  padding: 0.75rem;
  border: 2px solid #e5e7eb;
  border-radius: 8px;
  font-size: 1rem;
  transition: border-color 0.2s;
}

input:focus {
  outline: none;
  border-color: #3b82f6;
}

button {
  width: 100%;
  padding: 1rem;
  background: linear-gradient(135deg, #3b82f6, #1d4ed8);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

button:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
}

button:disabled {
  opacity: 0.7;
  cursor: not-allowed;
  transform: none;
}

.result {
  background: linear-gradient(135deg, #f0fdf4, #dcfce7);
  padding: 2rem;
  border-radius: 12px;
  border: 2px solid #22c55e;
}

.metrics {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.metric {
  text-align: center;
  padding: 1rem;
  background: white;
  border-radius: 8px;
}

.label {
  display: block;
  font-size: 0.875rem;
  color: #6b7280;
  margin-bottom: 0.25rem;
}

.value {
  font-size: 1.5rem;
  font-weight: bold;
  color: #059669;
}

.agents {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
  margin-top: 1rem;
}

.agent {
  padding: 1rem;
  border-radius: 8px;
  text-align: center;
  font-weight: 500;
}

.agent.budget {
  background: linear-gradient(135deg, #fee2e2, #fecaca);
  color: #dc2626;
}

.agent.compat {
  background: linear-gradient(135deg, #dbeafe, #bfdbfe);
  color: #2563eb;
}

.agent.profile {
  background: linear-gradient(135deg, #f3e8ff, #e9d5ff);
  color: #7c3aed;
}

.error {
  background: #fee2e2;
  color: #dc2626;
  padding: 1rem;
  border-radius: 8px;
  border-left: 4px solid #dc2626;
}
</style>
