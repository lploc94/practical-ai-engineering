from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
from presidio_analyzer.nlp_engine import NoOpNlpEngine
from presidio_analyzer.predefined_recognizers import CreditCardRecognizer, EmailRecognizer, PhoneRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

LANG = "vi"


def build_analyzer() -> AnalyzerEngine:
    """Analyzer chỉ dùng recognizer dạng pattern, không cần model spaCy cho tiếng Việt.
    Không nhận diện tên người và địa chỉ: việc đó cần model NER cho tiếng Việt."""
    registry = RecognizerRegistry(supported_languages=[LANG])
    registry.add_recognizer(EmailRecognizer(supported_language=LANG))
    registry.add_recognizer(CreditCardRecognizer(supported_language=LANG))
    # Mặc định PhoneRecognizer chỉ quét US, GB, DE, FR, IL, IN, CA, BR: phải thêm "VN".
    registry.add_recognizer(PhoneRecognizer(supported_language=LANG, supported_regions=("VN",)))
    # Số định danh cá nhân (số CCCD, số thẻ căn cước): 12 chữ số liền nhau.
    registry.add_recognizer(PatternRecognizer(
        supported_entity="VN_PERSONAL_ID", supported_language=LANG,
        patterns=[Pattern("12 chữ số", r"(?<!\d)\d{12}(?!\d)", 0.5)]))
    nlp = NoOpNlpEngine(models=[{"lang_code": LANG, "model_name": "no_op"}])
    return AnalyzerEngine(registry=registry, nlp_engine=nlp, supported_languages=[LANG])


ANALYZER = build_analyzer()
ANONYMIZER = AnonymizerEngine()


def redact(text: str, min_score: float = 0.4) -> tuple[str, list[str]]:
    """Thay PII bằng nhãn <LOẠI>; trả văn bản đã che và danh sách loại đã gặp."""
    found = ANALYZER.analyze(text=text, language=LANG, score_threshold=min_score)
    out = ANONYMIZER.anonymize(
        text=text, analyzer_results=found,
        operators={"DEFAULT": OperatorConfig("replace", {"new_value": "<PII>"}),
                   **{e: OperatorConfig("replace", {"new_value": f"<{e}>"})
                      for e in ("EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "VN_PERSONAL_ID")}})
    return out.text, sorted({r.entity_type for r in found})
