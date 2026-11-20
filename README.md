# FocusGuard

Monitor inteligente multimodal de foco para estudos de concurso público com **visão computacional local**, **dedução semântica de tela via Gemini** (sem gaiola de whitelist), **Coach Firme e Direto** e **interação completa por voz (microfone + TTS PT-BR)**.

## Início Rápido (Windows)

1. Copie `.env.example` para `.env` e adicione sua `GEMINI_API_KEY` (ou cole diretamente no Dashboard ao iniciar).
   - **Usa Google AI Pro?** Vincule seus **US$ 10/mês de créditos inclusos** no [Google AI Studio](https://aistudio.google.com/apikey) para rodar no Paid Tier com privacidade total sem custo extra.
2. Execute:
   ```powershell
   .\start.ps1
   ```
3. O dashboard abre em `http://127.0.0.1:8765`.

## Atalhos Globais
- `Ctrl + Shift + Space`: Falar com o Coach por voz (Push-to-Talk com detecção de silêncio)
- `Ctrl + Shift + P`: Modo Privacidade (desliga câmera e captura instantaneamente)
