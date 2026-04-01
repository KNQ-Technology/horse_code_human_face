<script setup lang="ts">
import { FileVideo } from 'lucide-vue-next';
import { useRoute } from 'vue-router';

const route = useRoute();
</script>

<template>
  <div class="app-layout">
    <header class="app-header">
      <div class="header-content">
        <div class="logo">
          <FileVideo class="logo-icon" :size="24" />
          <h1>赛马识别系统</h1>
        </div>
        <nav class="nav-tabs">
          <router-link to="/" class="nav-tab" :class="{ active: route.path === '/' }">
            完整分析
          </router-link>
          <router-link to="/simple" class="nav-tab" :class="{ active: route.path === '/simple' }">
            鞍垫号码
          </router-link>
        </nav>
      </div>
    </header>

    <main class="main-container">
      <router-view />
    </main>
  </div>
</template>

<style scoped>
.app-layout {
  min-height: 100vh;
  background-color: #090a0f;
  color: #e2e8f0;
  display: flex;
  flex-direction: column;
}

.app-header {
  height: 64px;
  background-color: rgba(15, 17, 26, 0.8);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid #1e293b;
  position: sticky;
  top: 0;
  z-index: 100;
}

.header-content {
  max-width: 1400px;
  margin: 0 auto;
  height: 100%;
  padding: 0 2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.logo {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.logo-icon {
  color: #6366f1;
}

.logo h1 {
  font-size: 1.25rem;
  font-weight: 700;
  margin: 0;
  background: linear-gradient(to right, #fff, #94a3b8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.nav-tabs {
  display: flex;
  gap: 0.25rem;
  background-color: #1e293b;
  padding: 4px;
  border-radius: 10px;
  border: 1px solid #334155;
}

.nav-tab {
  padding: 6px 18px;
  border: none;
  background: transparent;
  color: #94a3b8;
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.85rem;
  font-weight: 600;
  text-decoration: none;
  transition: all 0.2s;
}

.nav-tab.active {
  background-color: #6366f1;
  color: #fff;
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

.nav-tab:not(.active):hover {
  color: #f1f5f9;
  background-color: rgba(51, 65, 85, 0.4);
}

.main-container {
  flex: 1;
  display: flex;
  flex-direction: column;
}
</style>
