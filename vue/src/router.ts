import { createRouter, createWebHistory } from 'vue-router'
import FullAnalysis from './views/FullAnalysis.vue'
import SimpleAnalysis from './views/SimpleAnalysis.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: FullAnalysis },
    { path: '/simple', component: SimpleAnalysis },
  ],
})

export default router
