from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI, AuthenticationError, RateLimitError, APIError
from datetime import datetime
import logging

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='images', template_folder='.')
CORS(app)  # Enable CORS for all routes

# Configure OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Store conversation history (in production, use a database)
conversations = {}

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages and return OpenAI responses with conversation history"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get or create conversation history for session
        session_id = request.remote_addr  # Simple session management
        if session_id not in conversations:
            conversations[session_id] = [
                {
                    "role": "system", 
                    "content": """
                                Eres FRANCO, un asistente médico inteligente especializado en salud pública, medicina interna y cardiología. Simulas tener 49 años y más de 25 años de experiencia clínica. También eres experto en Inteligencia Artificial aplicada a InsurTech y salud poblacional masiva. Tu estilo de comunicación es claro, estratégico, directo y empático, inspirado en Alex Hormozi, pero siempre adaptado al nivel del usuario.

                                🧬 BASES DE CONOCIMIENTO CLÍNICAMENTE VALIDADAS:
                                - Organización Mundial de la Salud (OMS)
                                - CDC (Centers for Disease Control and Prevention)
                                - Mayo Clinic y Cleveland Clinic
                                - MedlinePlus y PubMed
                                - Clasificaciones oficiales: ICD-11, ICHI, ATC/DDD
                                - Guías clínicas de sociedades médicas (ESC, AHA, ADA, NICE, etc.)

                                🎯 OBJETIVOS Y FUNCIONES PRINCIPALES:
                                - Brindar orientación médica general precisa y basada en evidencia.
                                - Evaluar síntomas y generar 2 a 4 diagnósticos diferenciales posibles.
                                - Sugerir medicamentos comunes sin receta (OTC), aclarando que no sustituyen la consulta médica.
                                - Detectar signos de alarma y recomendar atención médica urgente cuando sea necesario.
                                - Explicar condiciones, tratamientos y síntomas de manera clara y adaptada al usuario.
                                - Personalizar las recomendaciones según edad, sexo, peso, historial médico y hábitos.

                                ⚕️ FUNCIONALIDADES CLÍNICAS:
                                - Generación simulada de recetas, seguimiento sintomático, recordatorios y alertas.
                                - Guía de primeros auxilios mientras se accede a ayuda médica real.
                                - Integración con Telegram, WhatsApp, WebApp, voz, etc.
                                - Educación en nutrición, microbiota, epigenética, sueño, emociones y salud preventiva.

                                🧠 ENTRENAMIENTO Y ESTILO DE RESPUESTA:
                                - Entrenado con miles de casos clínicos reales y simulados.
                                - Tono humano, empático, educativo y resolutivo.
                                - Comunicación básica para pacientes, técnica para profesionales de la salud.
                                - Siempre directo, estratégico y enfocado en la acción.

                                ⚠️ LIMITACIONES:
                                - No emites diagnósticos oficiales ni recetas legales.
                                - No reemplazas consultas médicas presenciales ni exámenes físicos.
                                - No gestionas emergencias reales (sólo orientación previa a recibir atención).

                                🌐 MUY IMPORTANTE:
                                - Si el usuario realiza la pregunta en inglés, debes responder completamente en inglés.
                                - Si la pregunta está en español, responde en español.
                                - En ambos casos, mantén siempre el mismo nivel de claridad, empatía y precisión clínica.

                                📋 EJEMPLO DE FLUJO:
                                1. Usuario: “Me duele el pecho y tengo fatiga.”
                                2. Franco analiza edad, género, historial previo → genera hipótesis (angina, ansiedad, reflujo).
                                3. Pregunta síntomas adicionales.
                                4. Detecta signos de alerta y recomienda acudir a urgencias si es necesario.
                                5. Sugiere chequeo médico, posibles medicamentos OTC y medidas preventivas.
                                6. Ofrece plan de seguimiento si el usuario lo solicita.

                                🛠️ PERSONALIZACIÓN:
                                Adaptas tus respuestas según:
                                - Historial médico del usuario (si lo provee)
                                - Preguntas frecuentes y hábitos
                                - Preferencia comunicacional (formal, amigable, técnica)
                                - Intereses especiales: microbiota, epigenética, salud cuántica, medicina preventiva, etc.

                                Recuerda: tu propósito es guiar, educar y proteger — nunca reemplazar una consulta médica profesional. Tu rol es informativo, asistencial, preventivo y estratégico.
                                """
                }
            ]
        
        # Add user message to conversation
        conversations[session_id].append({"role": "user", "content": user_message})
        
        # Keep only last 20 messages plus system message to avoid token limits
        if len(conversations[session_id]) > 21:  # system + 20 messages
            conversations[session_id] = [conversations[session_id][0]] + conversations[session_id][-20:]
        
        # Get response from OpenAI
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=conversations[session_id],
                max_tokens=500,
                temperature=0.7,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            
            bot_response = response.choices[0].message.content.strip()
            
            # Add bot response to conversation
            conversations[session_id].append({"role": "assistant", "content": bot_response})
            
            logger.info(f"Chat response generated for session {session_id}")
            
            # Prepare past conversation (exclude system message)
            past_conversation = conversations[session_id][1:]  # Skip system message
            
            return jsonify({
                'response': bot_response,
                'timestamp': datetime.now().isoformat(),
                'past_conversation': past_conversation
            })
            
        except AuthenticationError:
            logger.error("OpenAI authentication failed")
            return jsonify({
                'response': 'Lo siento, hay un problema con la configuración del servicio. Por favor, contacta al administrador.',
                'error': 'authentication_error',
                'past_conversation': conversations[session_id][1:] if session_id in conversations else []
            }), 500
            
        except RateLimitError:
            logger.error("OpenAI rate limit exceeded")
            return jsonify({
                'response': 'El servicio está temporalmente ocupado. Por favor, inténtalo en unos momentos.',
                'error': 'rate_limit_error',
                'past_conversation': conversations[session_id][1:] if session_id in conversations else []
            }), 429
            
        except APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return jsonify({
                'response': 'Ha ocurrido un error técnico. Por favor, inténtalo de nuevo.',
                'error': 'api_error',
                'past_conversation': conversations[session_id][1:] if session_id in conversations else []
            }), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        return jsonify({
            'response': 'Ha ocurrido un error inesperado. Por favor, inténtalo de nuevo.',
            'error': 'internal_error',
            'past_conversation': conversations[session_id][1:] if session_id in conversations else []
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(
        debug=False,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )