# AGENTS.md — Regras de Engenharia do FocusGuard

Este repositório é construído em paralelo por múltiplos agentes sob contratos congelados (`src/focus_guard/core/`).

## Regras Obrigatórias para Todos os Agentes
1. **NUNCA modifique arquivos fora da sua pasta de ownership**, especialmente `src/focus_guard/core/` e `tests/fakes/`.
2. **Privacidade First:** NUNCA salve screenshots ou frames de webcam em disco. Tudo fica em memória RAM efêmera.
3. **Resiliência:** Toda integração com hardware (webcam, microfone, Win32, UIA, rede, API Gemini) deve tratar falhas graciosamente com logging estruturado e degradação segura.
4. **Zero-Prison Philosophy:** NÃO crie whitelists rígidas que bloqueiem apps multiuso (Antigravity, Obsidian, Gemini/NotebookLM, YouTube). A decisão sobre conteúdo na tela pertence ao fluxo semântico.
5. **Personalidade Padrão:** "Coach Firme e Direto" — puxão de orelha incisivo baseado em fatos reais da tela/câmera, sem blá-blá-blá motivacional genérico.
6. **Qualidade (Definition of Done):** Antes de encerrar, execute usando o Python da `.venv`:
   - `.\.venv\Scripts\python.exe -m pytest <seu_arquivo_de_teste>`
   - `.\.venv\Scripts\python.exe -m ruff check <sua_pasta>`
   Todos os testes e lint devem passar com 0 erros.
