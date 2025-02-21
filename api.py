from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import re
import os
from difflib import SequenceMatcher
import anthropic
from openai import OpenAI
import asyncio
import json
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ThemisCode API", description="API for generating and analyzing educational questions")

# Question Tracker class
class QuestionTracker:
    def __init__(self, similarity_threshold=0.8):
        self.questions = []
        self.question_hashes = set()
        self.similarity_threshold = similarity_threshold
    
    def hash_question(self, question):
        """Creates a unique hash value for the question"""
        normalized = re.sub(r'\s+', ' ', question.lower().strip())
        return hash(normalized)
    
    def is_similar(self, new_question):
        """Checks similarity of new question to existing ones"""
        for existing in self.questions:
            similarity = SequenceMatcher(None, 
                                      new_question['question'], 
                                      existing['question']).ratio()
            if similarity > self.similarity_threshold:
                return True
        return False
    
    def add_question(self, question_data):
        """Adds new question and checks similarity"""
        question_hash = self.hash_question(question_data['question'])
        
        if question_hash not in self.question_hashes and not self.is_similar(question_data):
            self.questions.append(question_data)
            self.question_hashes.add(question_hash)
            return True
        return False

# Input models
class GenerateQuestionsRequest(BaseModel):
    main_topic: str
    sub_topics: List[str]
    content: str
    question_type: str
    question_count: int
    difficulty: int
    similarity_threshold: float = 0.70 #Sabit
    model: str
    model_temperature: float = 0.3
    openai_api_key: Optional[str] = "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"
    google_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = "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"

class AnalyzeQuestionRequest(BaseModel):
    question_text: str
    anthropic_api_key: str = "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"

# Response models
class QuestionData(BaseModel):
    question: str
    choices: str
    correct_answer: str
    explanation: str

class GenerateQuestionsResponse(BaseModel):
    success: bool
    total_generated: int
    questions: List[QuestionData]
    message: str

class AnalyzeQuestionResponse(BaseModel):
    score: int
    analysis: str

# Prompt generation functions
def generate_multiple_questions_prompt(main_topic, sub_topics, content, question_type, generated_count, total_count, difficulty):
    """Enhanced prompt for generating unique questions"""
    prompt = f"""
    Lütfen aşağıdaki bilgileri kullanarak {total_count} sorudan {generated_count+1}. soruyu Türkçe oluştur:
    Ana Konu: {main_topic}
    Alt Başlıklar: {', '.join(sub_topics)}
    İçerik: {content}
    Soru Türü: {question_type}
    Zorluk Seviyesi: %{difficulty}
    
    Önemli Kurallar:
    1. Şu ana kadar {generated_count} adet soru üretildi, şimdi TAM OLARAK BENZERSİZ bir {generated_count+1}. soru üretmelisin.
    2. Her soru birbirinden tamamen farklı olmalıdır - farklı alt konulara odaklanmalıdır.
    3. Daha önce üretilen sorulardan TAMAMEN farklı bir soru üret.
    4. Her soru için aşağıdaki formatı kullan:
    
    Soru: [Soru metni]
    Şıklar: (Eğer çoktan seçmeliyse)
    A) ...
    B) ...
    C) ...
    D) ...
    E) ...
    
    Doğru Cevap: [Cevap]
    
    Açıklama: [Detaylı açıklama]
    
    ZORUNLU KONTROLLER:
    - Tamamen farklı bir konsepte odaklanmalısın
    - Aynı kavramları sormaktan kaçınmalısın
    - Şıklar ve açıklamalar benzersiz olmalıdır
    - Her yeni soru öncekilerden belirgin şekilde farklı olmalıdır
    """
    return prompt

def generate_prompt(main_topic, sub_topics, content, question_type, question_count, difficulty):
    """Original prompt function preserved"""
    prompt = f"""
    Lütfen aşağıdaki bilgileri kullanarak Türkçe soru oluştur:\n
    Ana Konu: {main_topic}\n
    Alt Başlıklar: {', '.join(sub_topics)}\n
    İçerik: {content}\n
    Soru Türü: {question_type}\n
    Zorluk Seviyesi: %{difficulty}\n
    
    Aşağıdaki kurallara mutlaka uyunuz:\n
    1. Sorunun başına her zaman "Soru:" ekleyin.\n
    2. Eğer şıklar varsa, "Şıklar:" ifadesi ile her bir şıkkı alt alta sıralayın.\n
    3. Şıkların ardından "Doğru Cevap:" ifadesiyle doğru cevabı belirtin.\n
    4. En sona "Açıklama:" ekleyerek doğru cevabın neden doğru olduğunu açıklayın.\n
    5. Şıklar arasında ve diğer bölümler arasında **her zaman birer boş satır bırakın.**\n
    6. Her bir şık ayrı bir satırda yer almalıdır.\n
    7. Her bir şıkdan sonra **her zaman birer boş satır bırakın.**\n
    8. Her başlık ve içeriğinden sonra **her zaman birer boş satır bırakın.**\n
    9. Her "Açıklama:" yazımında sonra **her zaman bir boş satır bırakın.**.\n
    
    **Önemli Notlar:**
    - Üretilen sorular, içerik ve şıklar bakımından büyük farklılıklar içermelidir.
    - Herzaman 1 adet Soru: başlıklı soru üreteceksin asla birden fazla soru üretemezsin.
    - üretilen sorular her şartta herzaman birbirinden farklı akış anlam ve içeriklere sahip olmalıdır.
    """
    return prompt

def extract_question_details(response_text):
    """Extracts question, choices, correct answer and explanation from the response text."""
    question_match = re.search(r"Soru:\s*(.*?)\n\n", response_text, re.DOTALL)
    choices_match = re.findall(r"([A-E])\)\s*(.*?)\n", response_text)
    answer_match = re.search(r"Doğru Cevap:\s*(.*?)\n\n", response_text, re.DOTALL)
    explanation_match = re.search(r"Açıklama:\s*(.*)", response_text, re.DOTALL)

    question = question_match.group(1).strip() if question_match else "Soru bulunamadı."
    choices = "\n".join([f"{choice[0]}) {choice[1]}" for choice in choices_match]) if choices_match else "Şıklar bulunamadı."
    correct_answer = answer_match.group(1).strip() if answer_match else "Doğru cevap bulunamadı."
    explanation = explanation_match.group(1).strip() if explanation_match else "Açıklama bulunamadı."
    
    return question, choices, correct_answer, explanation

async def generate_llm_response(model_params, model_type, api_key, prompt, question_tracker):
    """Generate a response from the LLM and handle retries for similar questions"""
    max_regeneration_attempts = 10
    current_attempt = 0
    
    while current_attempt < max_regeneration_attempts:
        response_message = ""
        
        try:
            # Generate response based on model type
            if model_type == "openai":
                client = OpenAI(api_key=api_key)
                messages = [{"role": "user", "content": prompt}]
                
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=model_params["model"],
                    messages=messages,
                    temperature=model_params["temperature"],
                    max_tokens=4096
                )
                response_message = response.choices[0].message.content
                
            elif model_type == "google":
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(
                    model_name=model_params["model"],
                    generation_config={
                        "temperature": model_params["temperature"],
                    },
                )
                gemini_messages = [{"role": "user", "parts": [prompt]}]
                
                response = await asyncio.to_thread(
                    model.generate_content,
                    contents=gemini_messages
                )
                response_message = response.text
                
            elif model_type == "anthropic":
                client = anthropic.Anthropic(api_key=api_key)
                
                response = await asyncio.to_thread(
                    client.messages.create,
                    model=model_params["model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=model_params["temperature"],
                    max_tokens=4096
                )
                response_message = response.content[0].text
            
            # Extract question details
            question, choices, correct_answer, explanation = extract_question_details(response_message)
            
            # Create question data
            question_data = {
                "question": question,
                "choices": choices,
                "correct_answer": correct_answer,
                "explanation": explanation
            }
            
            # Check if question is unique
            if question_tracker.add_question(question_data):
                return True, question_data, None
            
            # Question was similar, retry
            current_attempt += 1
            if "temperature" in model_params:
                model_params["temperature"] = min(2.0, model_params["temperature"] + 0.1)
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return False, None, f"API error: {str(e)}"
            
    return False, None, "Maksimum yeniden oluşturma denemesi aşıldı."

async def analyze_question_with_claude(question_text, anthropic_api_key):
    """Analyzes the question using Claude for accuracy and logic"""
    prompt = f"""
    Aşağıda bir soru, doğru cevap ve açıklaması verilmiştir. Lütfen sorunun doğruluğunu ve mantıklılığını değerlendirerek 100 üzerinden bir puan verin.

    **Soru:**  
    {question_text}

    **İnceleme Kriterleri:**  
    - Sorunun konuya uygunluğu ve netliği 
    - Doğru cevabın mantıklı olup olmadığı 
    - Açıklamanın yeterli ve anlaşılır olması 

    Lütfen sadece aşağıdaki formatta bir çıktı üretin:
    - "Puan: X/100"
    - "Açıklama: [Kısa ve öz analiz]"
    """

    try:
        client = anthropic.Anthropic(api_key=anthropic_api_key)
        response = await asyncio.to_thread(
            client.messages.create,
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400
        )

        response_text = response.content[0].text

        # Extract score using regex
        score_match = re.search(r"Puan:\s*(\d+)/100", response_text)
        score = int(score_match.group(1)) if score_match else 0

        return score, response_text
    except Exception as e:
        logger.error(f"Error analyzing question: {str(e)}")
        return 0, f"Analiz sırasında hata oluştu: {str(e)}"

@app.post("/generate-questions", response_model=GenerateQuestionsResponse)
async def generate_questions(request: GenerateQuestionsRequest):
    # Validate API keys
    if (request.model.startswith("gpt") and not request.openai_api_key) or \
       (request.model.startswith("gemini") and not request.google_api_key) or \
       (request.model.startswith("claude") and not request.anthropic_api_key):
        raise HTTPException(status_code=400, detail="Selected model requires appropriate API key")
    
    # Determine model type
    model_type = None
    api_key = None
    if request.model.startswith("gpt"):
        model_type = "openai"
        api_key = request.openai_api_key
    elif request.model.startswith("gemini"):
        model_type = "google"
        api_key = request.google_api_key
    elif request.model.startswith("claude"):
        model_type = "anthropic"
        api_key = request.anthropic_api_key
    
    model_params = {
        "model": request.model,
        "temperature": request.model_temperature,
    }
    
    # Initialize question tracker
    question_tracker = QuestionTracker(similarity_threshold=request.similarity_threshold)
    
    # Start generating questions
    successful_questions = []
    total_successful = 0
    max_attempts = request.question_count * 2
    attempts = 0
    
    while total_successful < request.question_count and attempts < max_attempts:
        try:
            if attempts > 0:
                # Use enhanced prompt for diversity
                prompt = generate_multiple_questions_prompt(
                    request.main_topic,
                    request.sub_topics,
                    request.content,
                    request.question_type,
                    total_successful,
                    request.question_count,
                    request.difficulty
                )
            else:
                # First question uses standard prompt
                prompt = generate_prompt(
                    request.main_topic, 
                    request.sub_topics, 
                    request.content, 
                    request.question_type, 
                    request.question_count, 
                    request.difficulty
                )
            
            success, question_data, error_message = await generate_llm_response(
                model_params=model_params,
                model_type=model_type,
                api_key=api_key,
                prompt=prompt,
                question_tracker=question_tracker
            )
            
            if success and question_data:
                successful_questions.append(QuestionData(
                    question=question_data["question"],
                    choices=question_data["choices"],
                    correct_answer=question_data["correct_answer"],
                    explanation=question_data["explanation"]
                ))
                total_successful += 1
            
            attempts += 1
            
        except Exception as e:
            logger.error(f"Error in question generation: {str(e)}")
            return GenerateQuestionsResponse(
                success=False,
                total_generated=total_successful,
                questions=successful_questions,
                message=f"Error occurred: {str(e)}"
            )
    
    # Prepare response message
    if total_successful >= request.question_count:
        message = f"{request.question_count} soru başarıyla oluşturuldu!"
        success = True
    else:
        message = f"{total_successful}/{request.question_count} soru oluşturulabildi."
        success = total_successful > 0
    
    return GenerateQuestionsResponse(
        success=success,
        total_generated=total_successful,
        questions=successful_questions,
        message=message
    )

@app.post("/analyze-question", response_model=AnalyzeQuestionResponse)
async def analyze_question(request: AnalyzeQuestionRequest):
    if not request.anthropic_api_key:
        raise HTTPException(status_code=400, detail="Anthropic API key is required for analysis")
    
    try:
        score, analysis = await analyze_question_with_claude(
            request.question_text,
            request.anthropic_api_key
        )
        
        return AnalyzeQuestionResponse(
            score=score,
            analysis=analysis
        )
    except Exception as e:
        logger.error(f"Error in question analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze question: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)