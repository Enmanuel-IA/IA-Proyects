# IA-Proyects

AI Engineering Portfolio

Este repositorio consolida una colección de proyectos de Inteligencia Artificial aplicados a problemas reales dentro del desarrollo de software moderno. Cada proyecto ha sido diseñado bajo principios de AI Engineering, combinando modelos de machine learning, arquitecturas cloud-native y patrones orientados a producción.

El objetivo principal es demostrar la capacidad de diseñar, implementar y operar sistemas inteligentes end-to-end, integrando múltiples tecnologías en pipelines robustos, escalables y mantenibles.

Áreas de especialización
Computer Vision (CNN, Transfer Learning)
Natural Language Processing (NLP)
Sistemas basados en agentes (AI Agents)
Automatización de QA con IA
Integración con servicios cloud (Azure AI)
Seguridad y validación en sistemas LLM
Proyectos

1. Image Classification (CNN desde cero)

Descripción

Implementación de una red neuronal convolucional entrenada desde cero para clasificación de imágenes (cats vs dogs), enfocada en comprender el aprendizaje profundo sin modelos preentrenados.

Arquitectura

Input → Convolutional Layers → Pooling → Flatten → Dense → Output

Detalles técnicos
Arquitectura CNN personalizada
Funciones de activación ReLU
Regularización con Dropout
Normalización de imágenes
Entrenamiento supervisado
Objetivo

Comprender la extracción de características y el comportamiento de redes convolucionales.

2. Transfer Learning vs Scratch Model

Descripción

Comparación entre un modelo entrenado desde cero y un modelo preentrenado (MobileNetV2).

Arquitectura

Input → Preprocessing → [CNN desde cero / MobileNetV2] → Dense Layers → Output

Detalles técnicos
Uso de MobileNetV2 preentrenado en ImageNet
Fine-tuning parcial
Evaluación con dataset compartido
Métricas: accuracy y loss
Resultados
Mejor generalización con transfer learning
Reducción del tiempo de entrenamiento
Convergencia más rápida
Objetivo

Analizar diferencias entre aprendizaje desde cero y reutilización de conocimiento.

3. Sentiment Analysis (NLP)

Descripción

Modelo de clasificación de texto para detectar sentimientos en tweets.

Pipeline

Texto → Limpieza → TF-IDF → Logistic Regression → Predicción

Detalles técnicos
Preprocesamiento con expresiones regulares
Eliminación de ruido (URLs, menciones, símbolos)
Vectorización con TF-IDF
Clasificador Logistic Regression
Pipeline con scikit-learn
Métricas
Accuracy
Precision
Recall
F1-score
Objetivo

Construir un pipeline NLP eficiente, interpretable y reproducible.

AI Agents y Automatización

4. AI Test Case Generator
Descripción

Sistema basado en agentes para automatizar la generación de casos de prueba y scripts a partir de historias de usuario.

Arquitectura

Historia de usuario → Agente generador de casos → Agente generador de scripts → Integración con Azure

Detalles técnicos
Prompt engineering avanzado
Generación estructurada de casos de prueba
Automatización del ciclo QA
Integración con Azure Logic Apps
Automatización

Detección de cambios de estado (Design → To Do) que dispara automáticamente el flujo de generación y ejecución.

Objetivo

Reducir intervención manual y mejorar la eficiencia del ciclo de pruebas.

5. Prompt Validator Agent

Descripción

Agente diseñado para validar prompts y configuraciones antes de su despliegue en producción.

Arquitectura

Archivos de configuración → Motor de validación → Análisis → Reporte

Validaciones
Sintaxis (JSON/YAML)
Validación contra esquemas
Detección de vulnerabilidades (prompt injection, data leakage)
Evaluación de calidad del prompt
Verificación de preparación para producción
Objetivo

Garantizar prompts seguros, consistentes y listos para entornos productivos.

6. Agents Validator

Descripción

Sistema de testing automatizado para validar el comportamiento de agentes conversacionales en runtime.

Arquitectura

Escenarios de prueba → Ejecutor → Agente → Motor de validación → Reporte

Tipos de pruebas
Funcionales
Casos límite
Boundary testing
Seguridad (jailbreak, prompt injection)
Seguridad de contenido
Resiliencia
Detalles técnicos
Escenarios definidos en YAML
Validación basada en reglas
Ejecución automatizada
Generación de reportes estructurados
Integración
Azure AI Foundry
CI/CD (Jenkins)

Objetivo

Asegurar que los agentes sean confiables, seguros y aptos para producción.

Experiencia en proyectos reales (no públicos)

Además de los proyectos incluidos en este repositorio, he trabajado en soluciones de IA aplicadas en entornos empresariales, enfocadas en automatización, validación y seguridad de sistemas basados en LLMs.

AI Test Automation con agentes
Generación automática de casos de prueba desde historias de usuario
Generación de scripts automatizados
Integración con Azure
Orquestación mediante Logic Apps

Impacto:

Reducción del trabajo manual en QA
Mejora de la trazabilidad
Automatización del flujo completo de pruebas
Prompt Validator Agent
Validación de prompts y configuraciones
Detección de vulnerabilidades de seguridad
Evaluación de calidad y claridad
Generación de reportes técnicos

Enfoque:

Seguridad en LLM
Validación previa a producción
Mejora continua de prompts
Agents Validator
Testing automatizado de agentes conversacionales
Validación de comportamiento esperado
Pruebas de seguridad y resiliencia
Integración en pipelines CI/CD

Tecnologías:

Python
Azure AI Foundry
Jenkins
Stack tecnológico
Python
TensorFlow / Keras
Scikit-learn
OpenCV
NumPy / Pandas
Azure AI Foundry
Azure OpenAI
Azure Cognitive Services
Azure Logic Apps
Jenkins
Conceptos aplicados
Machine Learning
Deep Learning
NLP
Transfer Learning
Multi-Agent Systems
Prompt Engineering
AI Safety
Automatización de testing
Arquitecturas basadas en eventos
Enfoque de ingeniería

Este portafolio sigue principios de:

Diseño modular de sistemas de IA
Separación de responsabilidades
Automatización end-to-end
Integración con servicios cloud
Validación y seguridad en sistemas LLM
Valor diferencial

Este portafolio demuestra experiencia en:

Desarrollo de sistemas de IA end-to-end
Integración de modelos en flujos reales
Diseño de agentes inteligentes
Seguridad en aplicaciones de IA
Automatización del ciclo de desarrollo
Autor

Juan Lopez
AI Engineer
