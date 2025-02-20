import re

def extract_explanation(response_text):
    """Açıklamayı 'Açıklama:' ifadesinden sonrası olarak ayıklar."""
    # Açıklama kısmını ayıklamak için regex
    explanation_match = re.search(r"Açıklama:\s*(.*)", response_text, re.DOTALL)
    
    # Eğer açıklama bulunursa, alınacak
    explanation = explanation_match.group(1).strip() if explanation_match else "Açıklama bulunamadı."
    
    return explanation

# Örnek metin
response_text = """
Soru: İcra takibinde borçlunun borcunu ödememesi durumunda, aşağıdaki haciz işlemlerinden hangisi öncelikli olarak uygulanır?

Şıklar:
A) Gayrimenkul haczi

B) Banka hesaplarının haczi

C) Maaş haczi

D) Taşınır mal haczi

E) Kıymetli evrak haczi

Doğru Cevap: B

Açıklama: İcra hukukunda, borçlunun malvarlığına yönelik haciz işlemlerinde "en az masraflı ve en kolay paraya çevrilebilir mal" prensibi uygulanır. Bu nedenle, öncelikle borçlunun banka hesaplarına haciz konulur çünkü bu hem masrafsızdır hem de doğrudan nakde çevrilebilir durumdadır. Diğer haciz türleri, banka hesaplarında yeterli bakiye bulunmaması durumunda sırasıyla değerlendirilir.

Soru: İcra takibinde borçlunun mal beyanında bulunmaması durumunda aşağıdaki yaptırımlardan hangisi uygulanır?

Şıklar:
A) Para cezası

B) Disiplin hapsi

C) Ağır hapis cezası

D) Kamu hizmetinden yasaklanma

E) Adli para cezası

Doğru Cevap: B

Açıklama: İcra ve İflas Kanunu'nun 337. maddesine göre, mal beyanında bulunmayan borçlu, alacaklının şikayeti üzerine, 3 ayı geçmemek üzere disiplin hapsi ile cezalandırılır. Bu, borçluyu mal beyanında bulunmaya zorlamak için öngörülmüş bir tazyik hapsi niteliğindedir. Borçlu, hapiste iken mal beyanında bulunursa derhal serbest bırakılır.
"""

# Açıklamaları ayıklama
explanations = re.findall(r"Açıklama:\s*(.*?)(?=\nSoru:|$)", response_text, re.DOTALL)

# Ayıklanan açıklamaları yazdırma
for idx, explanation in enumerate(explanations, 1):
    print(f"Açıklama {idx}:\n{explanation}\n")
