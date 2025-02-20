import streamlit as st
from openai import OpenAI
import dotenv
import os
import re
import anthropic
from difflib import SequenceMatcher

# Question Tracker class from your first file
class QuestionTracker:
    def __init__(self):
        self.questions = []
        self.question_hashes = set()
    
    def hash_question(self, question):
        """Sorunun benzersiz hash değerini oluşturur"""
        # Soru metnini normalize et
        normalized = re.sub(r'\s+', ' ', question.lower().strip())
        return hash(normalized)
    
    def is_similar(self, new_question, similarity_threshold=0.8):
        """Yeni sorunun mevcut sorulara benzerliğini kontrol eder"""
        
        for existing in self.questions:
            similarity = SequenceMatcher(None, 
                                       new_question['question'], 
                                       existing['question']).ratio()
            if similarity > similarity_threshold:
                return True
        return False
    
    def add_question(self, question_data):
        """Yeni soruyu ekler ve benzerlik kontrolü yapar"""
        question_hash = self.hash_question(question_data['question'])
        
        if question_hash not in self.question_hashes and not self.is_similar(question_data):
            self.questions.append(question_data)
            self.question_hashes.add(question_hash)
            return True
        return False

def generate_multiple_questions_prompt(main_topic, sub_topics, content, question_type, generated_count, total_count, difficulty):
    """Benzersiz sorular üretmek için geliştirilmiş prompt"""
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
    """Orijinal prompt fonksiyonu korundu"""
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
    """Verilen metinden soruyu, şıkları, doğru cevabı ve açıklamayı çıkarır."""
    # Soru metnini çekmek için regex
    question_match = re.search(r"Soru:\s*(.*?)\n\n", response_text, re.DOTALL)
    choices_match = re.findall(r"([A-E])\)\s*(.*?)\n", response_text)  # Şıkları ayıklamak için
    answer_match = re.search(r"Doğru Cevap:\s*(.*?)\n\n", response_text, re.DOTALL)
    explanation_match = re.search(r"Açıklama:\s*(.*)", response_text, re.DOTALL) # Daha sağlam açıklama regex'i

    question = question_match.group(1).strip() if question_match else "Soru bulunamadı."
    choices = "\n".join([f"{choice[0]}) {choice[1]}" for choice in choices_match]) if choices_match else "Şıklar bulunamadı."
    correct_answer = answer_match.group(1).strip() if answer_match else "Doğru cevap bulunamadı."
    explanation = explanation_match.group(1).strip() if explanation_match else "Açıklama bulunamadı."
    
    return question, choices, correct_answer, explanation

# ... (keep all the imports and other code the same until the stream_llm_response function)

def stream_llm_response(model_params, model_type=None, api_key=None, prompt=None, question_count=None):
    response_message = ""
    
    # QuestionTracker'ı session state'e ekle
    if "question_tracker" not in st.session_state:
        st.session_state.question_tracker = QuestionTracker()
    
    # Maximum regeneration attempts for similar questions
    max_regeneration_attempts = 10
    current_attempt = 0
    
    while current_attempt < max_regeneration_attempts:
        response_message = ""  # Reset response message for each attempt
        
        # Generate response from the selected model
        if model_type == "openai":
            client = OpenAI(api_key=api_key)
            messages = [{"role": "user", "content": prompt}]

            for chunk in client.chat.completions.create(
                model=model_params["model"] if "model" in model_params else "gpt-4o",
                messages=messages,
                temperature=model_params["temperature"] if "temperature" in model_params else 0.3,
                max_tokens=4096,
                stream=True,
            ):
                chunk_text = chunk.choices[0].delta.content or ""
                response_message += chunk_text
                yield chunk_text if current_attempt == 0 else ""  # Only show streaming for first attempt

        elif model_type == "google":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name=model_params["model"],
                generation_config={
                    "temperature": model_params["temperature"] if "temperature" in model_params else 0.3,
                },
            )
            gemini_messages = [{"role": "user", "parts": [prompt]}]

            for chunk in model.generate_content(
                contents=gemini_messages,
                stream=True,
            ):
                chunk_text = (chunk.text or "").strip()
                response_message += chunk_text
                yield chunk_text if current_attempt == 0 else ""

        elif model_type == "anthropic":
            client = anthropic.Anthropic(api_key=api_key)
        
            with client.messages.stream(
                model=model_params["model"] if "model" in model_params else "claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": prompt}],
                temperature=model_params["temperature"] if "temperature" in model_params else 0.3,
                max_tokens=4096,
            ) as stream:
                for text in stream.text_stream: 
                    response_message += text
                    yield text if current_attempt == 0 else ""
                
        # Extract question details
        question, choices, correct_answer, explanation = extract_question_details(response_message)
        
        if "questions" not in st.session_state:
            st.session_state.questions = []

        # Create question data
        question_data = {
            "question": question,
            "choices": choices,
            "correct_answer": correct_answer,
            "explanation": explanation
        }
        
        # Check if question is unique
        if st.session_state.question_tracker.add_question(question_data):
            st.session_state.questions.append(question_data)
            st.session_state.messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": response_message,
                    }
                ]})
            return True
            
        else:
            current_attempt += 1
            if current_attempt < max_regeneration_attempts:
                yield f"\n\n❌ Benzer soru tespit edildi. Yeniden oluşturuluyor (Deneme {current_attempt + 1}/{max_regeneration_attempts})..."
                # Increase temperature slightly for each retry to encourage more variety
                if "temperature" in model_params:
                    model_params["temperature"] = min(2.0, model_params["temperature"] + 0.1)
            else:
                yield "\n\n❌ Maksimum yeniden oluşturma denemesi aşıldı. Lütfen farklı bir soru için tekrar deneyin."
                return False
    
    return False

# Update the QuestionTracker class to include similarity threshold as an instance variable
class QuestionTracker:
    def __init__(self, similarity_threshold=0.8):
        self.questions = []
        self.question_hashes = set()
        self.similarity_threshold = similarity_threshold
    
    def hash_question(self, question):
        """Sorunun benzersiz hash değerini oluşturur"""
        normalized = re.sub(r'\s+', ' ', question.lower().strip())
        return hash(normalized)
    
    def is_similar(self, new_question):
        """Yeni sorunun mevcut sorulara benzerliğini kontrol eder"""
        for existing in self.questions:
            similarity = SequenceMatcher(None, 
                                       new_question['question'], 
                                       existing['question']).ratio()
            if similarity > self.similarity_threshold:
                return True
        return False
    
    def add_question(self, question_data):
        """Yeni soruyu ekler ve benzerlik kontrolü yapar"""
        question_hash = self.hash_question(question_data['question'])
        
        if question_hash not in self.question_hashes and not self.is_similar(question_data):
            self.questions.append(question_data)
            self.question_hashes.add(question_hash)
            return True
        return False



def main():
    # --- Page Config ---
    st.set_page_config(
        page_title="The ThemisCode",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # Initialize session state for messages and question tracker
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "question_tracker" not in st.session_state:
        st.session_state.question_tracker = QuestionTracker()

    # --- Header ---
    st.html("""<h1 style="text-align: center;">🤖 <i>The ThemisCode</i> </h1>""")
    with st.sidebar:
        st.subheader("Sayfa Seçimi")
        page = st.radio("Sayfa Seçin", ["Ana Sayfa", "Boş Sayfa"])
        
    if page == "Ana Sayfa":
        with st.sidebar:
            st.subheader("📌 Oluşturulan Sorular")
            if "questions" not in st.session_state:
                st.session_state.questions = []
            if st.session_state.questions:
                for idx, q in enumerate(st.session_state.questions):
                    with st.expander(f"🔹 Soru {idx+1}"):
                        st.write(f"**Soru:** {q['question']}")
                        st.write(f"**Şıklar:** {q['choices']}" if "Şıklar" not in q["choices"] else "")
                        st.write(f"**Doğru Cevap:** {q['correct_answer']}")
                        st.write(f"**Açıklama:** {q['explanation']}")
                        
        # --- Side Bar ---
        with st.sidebar:
            cols_keys = st.columns(2)
            with cols_keys[0]:
                default_openai_api_key = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") is not None else ""
                with st.popover("🔐 OpenAI"):
                    openai_api_key = st.text_input("Introduce your OpenAI API Key (https://platform.openai.com/)", value=default_openai_api_key, type="password")

            with cols_keys[1]:
                default_google_api_key = os.getenv("GOOGLE_API_KEY") if os.getenv("GOOGLE_API_KEY") is not None else ""
                with st.popover("🔐 Google"):
                    google_api_key = st.text_input("Introduce your Google API Key (https://aistudio.google.com/app/apikey)", value=default_google_api_key, type="password")

            default_anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") if os.getenv("ANTHROPIC_API_KEY") is not None else ""
            with st.popover("🔐 Anthropic"):
                anthropic_api_key = st.text_input("Introduce your Anthropic API Key (https://console.anthropic.com/)", value=default_anthropic_api_key, type="password")

        # --- Main Content ---
        if (openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key) and (google_api_key == "" or google_api_key is None) and (anthropic_api_key == "" or anthropic_api_key is None):
            st.write("#")
            st.warning("⬅️ Please introduce an API Key to continue...")
        else:
            # Displaying the previous messages if there are any
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    for content in message["content"]:
                        if content["type"] == "text":
                            st.write(content["text"])

            # Side bar model options and inputs
            with st.sidebar:
                st.divider()
                anthropic_models = ["claude-3-5-sonnet-20241022"]
                google_models = ["gemini-1.5-flash", "gemini-1.5-pro"]
                openai_models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"]
                
                available_models = [] + (anthropic_models if anthropic_api_key else []) + (google_models if google_api_key else []) + (openai_models if openai_api_key else [])
                model = st.selectbox("Select a model:", available_models, index=0)
                model_type = None
                if model.startswith("gpt"): model_type = "openai"
                elif model.startswith("gemini"): model_type = "google"
                elif model.startswith("claude"): model_type = "anthropic"
                
                main_topic = st.text_input("Konu Başlığı")
                sub_topics = st.text_area("Alt Başlıklar (Virgülle ayırın)").split(",")
                content = st.text_area("İçerik")
                question_type = st.selectbox("Soru Türü", ["Açık uçlu", "Çoktan seçmeli", "Doğru/Yanlış"])

                with st.popover("⚙️ Soru Ayarları"):
                    question_count = st.slider("Soru Sayısı", min_value=1, max_value=20, value=5, step=1)
                    difficulty = st.slider("Zorluk Seviyesi (%)", min_value=0, max_value=100, value=50, step=5)
                    similarity_threshold = st.slider("Benzerlik Eşiği", min_value=0.5, max_value=0.9, value=0.75, step=0.05, 
                                                    help="Daha düşük değerler daha katı benzersizlik kontrolü yapar")

                with st.popover("⚙️ Model Parametreleri"):
                    model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

                model_params = {
                    "model": model,
                    "temperature": model_temp,
                }

                def reset_conversation():
                    if "messages" in st.session_state and len(st.session_state.messages) > 0:
                        st.session_state.pop("messages", None)
                    if "questions" in st.session_state:
                        st.session_state.pop("questions", None)
                    if "question_tracker" in st.session_state:
                        st.session_state.pop("question_tracker", None)

                # Eğer onay durumu yoksa başlat
                if "confirm_reset" not in st.session_state:
                    st.session_state.confirm_reset = False

                # "Soruları Sıfırla" butonu
                if not st.session_state.confirm_reset:
                    if st.button("🗑️ Soruları Sıfırla"):
                        st.session_state.confirm_reset = True  # Onay ekranını aç

                # Onay ekranı
                if st.session_state.confirm_reset:
                    st.warning("❗ **Soruları silmek istediğinize emin misiniz?** Bu işlem geri alınamaz.")

                    # Butonlar yan yana olacak şekilde iki sütun
                    col1, col2 = st.columns(2)

                    # "Evet" butonu
                    with col1:
                        if st.button("✅ Evet, sil!"):
                            reset_conversation()
                            st.session_state.confirm_reset = False  # Onayı sıfırla
                            st.rerun()  # Sayfayı yenile

                    # "Vazgeç" butonu
                    with col2:
                        if st.button("❌ Vazgeç"):
                            st.session_state.confirm_reset = False  # Onayı iptal et
                            st.rerun()

            # Button to generate multiple questions
            if st.button("Soruları Oluştur", on_click=reset_conversation):
                # QuestionTracker benzerlik eşiğini güncelle
                if "question_tracker" in st.session_state:
                    # Dinamik olarak benzerlik eşiğini ayarla
                    st.session_state.question_tracker.similarity_threshold = similarity_threshold
                
                model2key = {
                    "openai": openai_api_key,
                    "google": google_api_key,
                    "anthropic": anthropic_api_key,
                }
                
                # Soruları üretmeye başla
                total_successful = 0
                max_attempts = question_count * 2  # Her soru için en fazla 2 deneme yapılabilir
                attempts = 0
                
                with st.status("Sorular oluşturuluyor...") as status:
                    while total_successful < question_count and attempts < max_attempts:
                        if attempts > 0:
                            # Gelişmiş prompt kullan - benzersiz soru üretmek için
                            promptx = generate_multiple_questions_prompt(
                                main_topic, 
                                sub_topics, 
                                content, 
                                question_type, 
                                total_successful,
                                question_count, 
                                difficulty
                            )
                        else:
                            # İlk soru için normal prompt
                            promptx = generate_prompt(main_topic, sub_topics, content, question_type, question_count, difficulty)
                        
                        with st.chat_message("assistant"):
                            success = st.write_stream(
                                stream_llm_response(
                                    model_params=model_params,
                                    model_type=model_type,
                                    api_key=model2key[model_type],
                                    prompt=promptx,
                                    question_count=question_count
                                )
                            )
                            
                            if success:
                                total_successful += 1
                                st.success(f"✅ Soru {total_successful}/{question_count} başarıyla oluşturuldu")
                            else:
                                st.warning(f"⚠️ Benzer soru tespit edildi, yeniden deneniyor...")
                        
                        attempts += 1
                    
                    if total_successful >= question_count:
                        status.update(label=f"✅ {question_count} soru başarıyla oluşturuldu!", state="complete")
                    else:
                        status.update(label=f"⚠️ {total_successful}/{question_count} soru oluşturulabildi.", state="complete")

           # --- Sidebar Analysis Section ---
            with st.sidebar:
                st.divider()
                st.subheader("📝 Claude Analiz Bölümü")

                # Eğer session_state içinde analiz sonuçları yoksa boş bir dictionary olarak başlat
                if "analyses" not in st.session_state:
                    st.session_state.analyses = {}

                if len(st.session_state.messages) > 0:
                    for i, message in enumerate(st.session_state.messages):
                        question_text = message["content"][0]["text"]
                        button_key = f"question_{i}"  # Her buton için benzersiz bir anahtar

                        if st.button(f"🔍 Soru {i+1} Analiz Et", key=button_key):
                            score, analysis = analyze_question_with_claude(question_text, anthropic_api_key)
                            st.session_state.analyses[button_key] = (score, analysis)  # Sonucu kaydet

                        # Eğer daha önce analiz edilmişse sonucu göster
                        if button_key in st.session_state.analyses:
                            score, analysis = st.session_state.analyses[button_key]

                            if score >= 70:
                                st.success(f"✅ **Puan: {score}/100**\n\n{analysis}")
                            else:
                                st.error(f"❌ **Puan: {score}/100**\n\n{analysis}")
                else:
                    st.info("Henüz analiz edilecek bir soru oluşturulmadı.")

    elif page == "Boş Sayfa":
          st.empty()  


# Analyze function from your code
def analyze_question_with_claude(question_text, anthropic_api_key):
    """Claude modelini kullanarak sorunun doğruluk ve mantıklık açısından analizini yapar"""

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

    client = anthropic.Anthropic(api_key=anthropic_api_key)
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200,
    )

    response_text = response.content[0].text

    # Puanı regex ile çekelim
    score_match = re.search(r"Puan:\s*(\d+)/100", response_text)
    score = int(score_match.group(1)) if score_match else 0

    return score, response_text

if __name__=="__main__":
    main()