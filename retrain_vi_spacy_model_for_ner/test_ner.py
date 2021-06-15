import spacy
MODEL_PATH = '/root/spacy_ner/models/vi_spacy_model_ner/model-best'
nlp = spacy.load(MODEL_PATH)

sample = """
Ca bệnh BN10949 ghi nhận tại tỉnh Lạng Sơn: nữ, 29 tuổi, địa chỉ tại huyện Bắc Sơn, tỉnh Lạng Sơn; liên quan đến Khu công nghiệp Đình Trám. Kết quả xét nghiệm ngày 15-6 dương tính với SARS-CoV-2. Hiện đang được cách ly, điều trị tại Trung tâm Y tế huyện Bắc Sơn, tỉnh Lạng Sơn.
"""

doc = nlp(sample)

for ent in doc.ents:
    print(ent.label_, ':', ent.text)