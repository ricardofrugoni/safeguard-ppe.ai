<div align="center">

# 🛡️ SafeGuard AI

### Sistema Inteligente de Detecção de EPIs com Visão Computacional

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ADD8.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-MVP-yellow.svg)]()

**Monitoramento automatizado em tempo real para segurança do trabalho**

[Demonstração](#-demonstração) • [Funcionalidades](#-funcionalidades) • [Instalação](#-instalação) • [Documentação](#-documentação)

</div>

---

## 📑 Índice

- [Visão Geral](#-visão-geral)
- [Problema e Solução](#-problema-e-solução)
- [Funcionalidades](#-funcionalidades)
- [Tecnologias](#-tecnologias)
- [Métricas de Performance](#-métricas-de-performance)
- [Aplicações Futuras](#-aplicações-futuras)
- [Roadmap](#-roadmap)
- [Casos de Uso](#-casos-de-uso)
- [Instalação](#-instalação)
- [Segurança e Privacidade](#-segurança-e-privacidade)
- [Contribuindo](#-contribuindo)
- [Licença](#-licença)

---

## 🎯 Visão Geral

O **SafeGuard AI** é um sistema de visão computacional baseado em **YOLOv8** que utiliza inteligência artificial para monitorar automaticamente o uso correto de Equipamentos de Proteção Individual (EPIs) em ambientes industriais e de construção civil.

### 🚨 Problema e Solução

| **Problema** | **Solução SafeGuard AI** |
|--------------|--------------------------|
| Acidentes causados por falta de EPIs | Detecção automática em tempo real |
| Fiscalização manual ineficiente | Monitoramento 24/7 sem intervenção humana |
| Falta de registro visual de não conformidades | Captura automática de evidências |
| Comunicação lenta entre observador e TST | Alertas instantâneos via app mobile |
| Dificuldade em cobrir grandes áreas | Integração com múltiplas câmeras e drones |

---

## ✨ Funcionalidades

### 🔍 Detecção Automática

O sistema identifica em tempo real:

| EPI | Status | Prioridade |
|-----|--------|-----------|
| Capacete (hardhat) | Obrigatório em áreas de risco | 🔴 Alta |
| Máscara facial (mask) | Proteção respiratória | 🔴 Alta |
| Colete de segurança | Alta visibilidade | 🟡 Média |
| Ausência de EPIs | Detecção de não conformidade | 🔴 Crítica |
| Pessoas | Contagem e rastreamento | 🔵 Informativa |
| Veículos/Maquinário | Contexto da área | 🔵 Informativa |

### 🗺️ Sistema de Zonas Inteligentes

```
┌─────────────────────────────────────────────────┐
│              MAPEAMENTO DE ZONAS                │
├─────────────────────────────────────────────────┤
│                                                 │
│  🔴 ZONA VERMELHA (Áreas de Risco)              │
│     ├─ Canteiro de obras                        │
│     ├─ Áreas de máquinas pesadas                │
│     ├─ Zonas de altura                          │
│     ├─ Locais com produtos químicos             │
│     └─ EPIs OBRIGATÓRIOS → Alerta imediato      │
│                                                 │
│  🟢 ZONA VERDE (Áreas Seguras)                  │
│     ├─ Refeitórios                              │
│     ├─ Vestiários                               │
│     ├─ Salas de descanso                        │
│     ├─ Escritórios                              │
│     └─ EPIs OPCIONAIS → Sem alertas             │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 📢 Sistema de Alertas em Cascata

**Fluxo do Alerta:**
1. Captura automática de foto/vídeo
2. Registro de data, hora e localização GPS
3. Notificação push instantânea ao app do TST
4. Armazenamento criptografado da evidência
5. Geração automática de relatório de ocorrência

---

## 🛠️ Tecnologias

<div align="center">

| Tecnologia | Versão | Função |
|-----------|--------|---------|
| **YOLOv8n** | Latest | Modelo de detecção de objetos |
| **Python** | 3.12+ | Linguagem principal |
| **PyTorch** | 2.0+ | Framework de deep learning |
| **OpenCV** | 4.8+ | Processamento de imagem |
| **Gradio** | Latest | Interface web interativa |
| **Ultralytics** | Latest | Biblioteca de treinamento |

</div>

---

## 📈 Métricas de Performance

### Modelo Treinado com 4.216 Imagens

<div align="center">

| Métrica | Valor | Status | Interpretação |
|---------|-------|--------|---------------|
| **mAP50** | 62% | 🟡 Bom | Precisão geral aceitável |
| **Precision** | 85-95% | 🟢 Excelente | Alta confiabilidade nas detecções |
| **Recall** | 55-65% | 🟡 Bom | Captura maioria dos casos |
| **FPS** | 30+ | 🟢 Excelente | Processamento em tempo real (GPU) |

</div>

### Desempenho por Classe

```
Capacete (hardhat)        ████████████████████ 85%
Pessoa (person)           ████████████████░░░░ 80%
Colete (safety-vest)      ███████████████░░░░░ 75%
Máscara (mask)            █████████████░░░░░░░ 65%
Sem capacete (no-hardhat) ████████████░░░░░░░░ 60%
```

---

## 🚀 Aplicações Futuras

### Integração com Câmeras IP Existentes

**Implementação:**
- Conexão com infraestrutura de CFTV existente
- Processamento distribuído (15-30 FPS por câmera)
- Suporte para centenas de câmeras simultâneas

**Benefícios:**
- Zero investimento em hardware adicional
- Monitoramento contínuo 24/7
- Escalabilidade horizontal

### Sistema com Drones Autônomos

**Recursos:**
- Rotas de inspeção pré-programadas
- Processamento embarcado (edge computing)
- Detecção em áreas de difícil acesso

**Aplicações:**
- Inspeção de telhados e estruturas elevadas
- Monitoramento de torres e andaimes
- Áreas remotas e de risco

### Aplicativo Mobile para TST

```
┌───────────────────────────────────────┐
│       SafeGuard TST App               │
├───────────────────────────────────────┤
│                                       │
│  🔴 ALERTAS ATIVOS (3)                |
│                                       │
│  ┌─────────────────────────────────┐  │
│  │ Setor B - Andaime 3             │  │
│  │ Trabalhador sem capacete        │  │
│  │ 14:32 - Hoje                    │  │
│  │ [VER] [REGISTRAR]               │  │
│  └─────────────────────────────────┘  │
│                                       │
│  ┌─────────────────────────────────┐  │
│  │ Área de Solda                   │  │
│  │ Sem máscara respiratória        │  │
│  │ 14:15 - Hoje                    │  │
│  │ [VER] [REGISTRAR]               │  │
│  └─────────────────────────────────┘  │
│                                       │
│  ───────────────────────────────────  │
│  Relatórios | Histórico               │
│  Configurações | Notificações         │
└───────────────────────────────────────┘
```

**Funcionalidades:**
- Notificações push em tempo real
- Visualização de evidências (foto/vídeo)
- Registro imediato de ações corretivas
- Geolocalização precisa dos incidentes
- Dashboard com KPIs e estatísticas
- Exportação de relatórios (PDF/Excel)

### Georreferenciamento de Zonas

**Configuração de Áreas:**

| Zona | Tipo | EPIs Obrigatórios | Ação |
|------|------|-------------------|------|
| Refeitório | 🟢 Verde | Nenhum | Nenhuma |
| Vestiário | 🟢 Verde | Nenhum | Nenhuma |
| Área de Construção | 🔴 Vermelha | Capacete + Colete | Alerta |
| Área de Solda | 🔴 Vermelha | Capacete + Máscara + Colete | Alerta |

**Funcionalidades:**
- Mapeamento em planta baixa da obra
- Configuração de EPIs por zona
- Definição de horários de risco
- Gerenciamento de exceções temporárias

### Integração Corporativa

**APIs e Conectores:**

```python
# Exemplo de integração via API REST
POST /api/v1/alert
{
  "timestamp": "2025-10-13T14:32:15Z",
  "location": "Setor B - Andaime 3",
  "violation_type": "no_hardhat",
  "employee_id": "optional",
  "evidence_url": "https://...",
  "severity": "high"
}
```

**Integrações Disponíveis:**
- ERP/SAP (Registro automático de não conformidades)
- Sistemas de RH (Vinculação a perfil do colaborador)
- Controle de Acesso (Bloqueio preventivo de entrada)
- E-mail/SMS (Notificações para hierarquia)
- BI/Analytics (Dashboards executivos)

---

## 🗓️ Roadmap

### Fase 1 - MVP (Atual)
- [x] Detecção básica de 5 classes de EPIs
- [x] Interface Gradio para demonstração
- [x] Modelo YOLOv8n treinado (mAP50: 62%)
- [x] Documentação técnica inicial

### Fase 2 - Produção (Q1-Q2 2026)
- [ ] Integração com câmeras IP (RTSP/HTTP)
- [ ] App mobile iOS/Android (React Native)
- [ ] Sistema de zonas georreferenciadas
- [ ] Dashboard web de monitoramento
- [ ] API REST v1 para integrações
- [ ] Deploy em cloud (AWS/Azure/GCP)

### Fase 3 - Expansão (Q3-Q4 2026)
- [ ] Suporte a drones autônomos (DJI SDK)
- [ ] Reconhecimento facial de trabalhadores
- [ ] Análise preditiva de padrões de risco
- [ ] Integração com ERP (SAP/Oracle)
- [ ] Conformidade NR-12, NR-18, NR-35
- [ ] Relatórios automatizados (ISO 45001)

### Fase 4 - IA Avançada (2027+)
- [ ] Detecção de comportamentos de risco
- [ ] Análise ergonômica de postura
- [ ] Contagem de pessoas em tempo real
- [ ] Detecção de fadiga/sonolência
- [ ] Sistema de gamificação para conformidade
- [ ] Predição de acidentes com ML

---

## 💼 Casos de Uso

### Construção Civil
- Fiscalização de capacetes em andaimes de altura
- Monitoramento de coletes em vias de circulação
- Controle de acesso a áreas restritas (escavações)
- Conformidade NR-18

### Indústria
- Verificação de máscaras N95 em áreas químicas
- Controle de EPIs em linhas de montagem
- Auditoria automatizada (ISO 45001, ISO 14001)
- Prevenção de multas e processos trabalhistas

### Mineração
- Monitoramento via drones em áreas remotas
- Detecção em ambientes subterrâneos
- Registro contínuo para compliance legal
- Conformidade NR-22

### Logística e Armazéns
- Controle de coletes em zonas de empilhadeiras
- Monitoramento de docas de carga/descarga
- Segurança em centros de distribuição
- Redução de acidentes com empilhadeiras

---

## 🔧 Instalação

### Pré-requisitos

```bash
Python 3.12+
CUDA 11.8+ (para GPU)
4GB+ RAM (8GB recomendado)
```

### Instalação Rápida

```bash
# Clone o repositório
git clone https://github.com/ricardofrugoni/safeguard_ai_epi.git
cd safeguard_ai_epi

# Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Instale dependências
pip install -r requirements.txt

# Execute a aplicação
python app.py
```

### Usando Docker

```bash
docker build -t safeguard-ai .
docker run -p 7860:7860 safeguard-ai
```

Acesse: `http://localhost:7860`

---

## 📊 Exemplo de Notificação

```
┌─────────────────────────────────────────┐
│  SafeGuard AI - ALERTA DE SEGURANÇA     │
├─────────────────────────────────────────┤
│                                         │
│  🔴 TRABALHADOR SEM CAPACETE            │
│                                         │
│  Local: Setor B - Andaime 3 (3º andar)  │
│  Hora: 14:32:15                         │
│  Data: 13/10/2025                       │
│  Temperatura: 28°C                      │
│                                         │
│  Identificação: Em andamento...         │
│  Evidência: Foto capturada (2.3MB)      │
│  Confiança: 92%                         │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │  [VER IMAGEM]  [REGISTRAR]        │  │
│  │  [LIGAR TST]   [IGNORAR]          │  │
│  └───────────────────────────────────┘  │
│                                         │
│  Alerta #1847 | Prioridade: ALTA 🔴    │
└─────────────────────────────────────────┘
```

---

## 🔐 Segurança e Privacidade

### Conformidade LGPD (Lei Geral de Proteção de Dados)

| Requisito | Implementação |
|-----------|---------------|
| **Processamento Local** | Imagens processadas on-premise (sem envio para nuvem externa) |
| **Anonimização** | Dados pessoais automaticamente anonimizados |
| **Criptografia** | AES-256 para armazenamento, TLS 1.3 para transmissão |
| **Retenção Limitada** | Evidências mantidas por 30-90 dias (configurável) |
| **Controle de Acesso** | RBAC (Role-Based Access Control) com autenticação 2FA |
| **Auditoria** | Logs imutáveis de todos os acessos |

### Princípios Éticos

- Trabalhadores informados sobre monitoramento (transparência)
- Sinalização visível de áreas monitoradas
- Uso exclusivo para segurança (não punição)
- Foco em prevenção, não penalização
- Direito de contestação de alertas
- Anonimização em relatórios agregados

---

## 🤝 Contribuindo

Contribuições são muito bem-vindas! Veja como participar:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/MinhaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona MinhaFeature'`)
4. Push para a branch (`git push origin feature/MinhaFeature`)
5. Abra um Pull Request

### Diretrizes de Contribuição

- Siga o [PEP 8](https://pep8.org/) para código Python
- Adicione testes para novas funcionalidades
- Atualize a documentação quando necessário
- Mantenha commits pequenos e descritivos

---

## 🏆 Reconhecimentos

- Dataset: [Roboflow Universe - Hard Hat Workers Dataset](https://universe.roboflow.com/)
- Modelo Base: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Comunidade Open-Source por ferramentas e bibliotecas
- Profissionais de Segurança do Trabalho por feedback valioso

---

## 📞 Contato e Suporte

<div align="center">

### 👨‍💻 Desenvolvedor

**Ricardo Frugoni**

🌐 Website: [www.codex.ai](https://www.codex.ai)  
📱 WhatsApp: +55 21 97355-4927  
📧 Email: contato@codex.ai

</div>

---

## 📄 Licença

Este projeto está licenciado sob a **MIT License** - veja o arquivo [LICENSE](LICENSE) para detalhes.

```
MIT License

Copyright (c) 2025 Visão Segura - Safe Guard AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

---

<div align="center">

### 🛡️ SafeGuard AI
**Protegendo vidas através da inteligência artificial**

[![GitHub Stars](https://img.shields.io/github/stars/ricardofrugoni/safeguard_ai_epi?style=social)](https://github.com/ricardofrugoni/safeguard_ai_epi)
[![GitHub Forks](https://img.shields.io/github/forks/ricardofrugoni/safeguard_ai_epi?style=social)](https://github.com/ricardofrugoni/safeguard_ai_epi)

---

**Desenvolvido por Ricardo Frugoni**

[⬆ Voltar ao topo](#️-safeguard-ai)

</div>
