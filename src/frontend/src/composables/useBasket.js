// frontend/src/composables/useBasket.js
// Теория: Composable = функция, возвращающая reactive state + methods
// Как useState/useEffect в React

import { ref, computed } from 'vue'

// Функция возвращает объект со всем state
export function useBasket() {
  // === STATE ===
  const userQuery = ref('')
  const basket = ref([])
  const loading = ref(false)
  const error = ref(null)
  const diet = ref('любая')
  const allergies = ref('')
  const originalPrice = ref(0)

  // === COMPUTED ===
  const totalPrice = computed(() => 
    basket.value.reduce((sum, item) => sum + (item.price || 0), 0)
  )

  const agentLabel = {
    budget: '💰 Бюджет',
    compatibility: '🔗 Совместимость',
    profile: '👤 Профиль'
  }

  // === METHODS ===
  async function optimizeBasket() {
    if (!userQuery.value.trim()) {
      error.value = '⚠️ Введите запрос!'
      basket.value = []
      return
    }

    loading.value = true
    error.value = null
    basket.value = []

    try {
      const response = await fetch('/api/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: userQuery.value,
          diet: diet.value,
          allergies: allergies.value
        })
      })

      if (!response.ok) {
        throw new Error(`Server error ${response.status}`)
      }

      const data = await response.json()
      basket.value = data.basket || []
      originalPrice.value = data.original_price || totalPrice.value * 1.2
    } catch (err) {
      error.value = `❌ Ошибка: ${err.message}`
    } finally {
      loading.value = false
    }
  }

  function formatPrice(price) {
    return new Intl.NumberFormat('ru-RU').format(Math.round(price))
  }

  function addToCart() {
    alert(`✅ Добавлено ${basket.value.length} товаров!`)
  }

  // Возвращаем ВСЁ, что нужно компоненту
  return {
    // State
    userQuery,
    basket,
    loading,
    error,
    diet,
    allergies,
    originalPrice,
    totalPrice,
    agentLabel,
    // Methods
    optimizeBasket,
    formatPrice,
    addToCart
  }
}
