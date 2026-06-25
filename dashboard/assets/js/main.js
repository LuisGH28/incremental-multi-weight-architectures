import { connectSSE } from './services/sse.js';
import { setupNavigation, setupLucideIcons } from './ui/navigation.js';

connectSSE();
setupNavigation();
setupLucideIcons();
