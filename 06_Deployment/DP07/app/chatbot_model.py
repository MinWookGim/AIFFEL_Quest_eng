"""
Day 7 - 한국어 챗봇 모델 (인스트럭션 튜닝 모델)
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class ChatbotModel:
    """Hugging Face 인스트럭트 모델을 로드하고 대화 응답을 생성합니다."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"          # Apple Silicon GPU
        else:
            self.device = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.model_name = model_name

    def generate_response(
        self,
        messages: list[dict],
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        """
        대화 기록을 받아 응답을 생성합니다.

        Args:
            messages: [{"role": "user", "content": "안녕"}, {"role": "bot", "content": "안녕하세요!"}, ...]
            max_new_tokens: 최대 생성 토큰 수
            temperature: 생성 다양성
        Returns:
            생성된 응답 텍스트
        """
        # 대화 기록 → 채팅 템플릿 (모델이 학습한 대화 형식 그대로)
        chat = [{"role": "system",
                 "content": "너는 친절한 한국어 챗봇이야. 반드시 한국어로, 두세 문장으로 자연스럽게 대답해."}]
        for msg in messages:
            role = "user" if msg["role"] == "user" else "assistant"
            chat.append({"role": role, "content": msg["content"]})

        def encode(chat_messages):
            """채팅 기록을 모델 입력 텐서로 만듭니다."""
            encoded = self.tokenizer.apply_chat_template(     # *your code* — 프롬프트 구성
                chat_messages, add_generation_prompt=True, return_tensors="pt"
            )
            # transformers 버전에 따라 Tensor 또는 BatchEncoding(dict)이 반환됩니다.
            # 최신 버전은 후자이므로 input_ids만 꺼내 텐서로 통일합니다.
            if not torch.is_tensor(encoded):
                encoded = encoded["input_ids"]
            return encoded.to(self.device)

        input_ids = encode(chat)

        # 토큰 예산 초과 시 오래된 턴부터 잘라낸다 (모델 최대 길이 초과 방지)
        max_length = getattr(self.model.config, "max_position_embeddings", 2048)
        while input_ids.shape[1] > max_length - max_new_tokens and len(chat) > 2:
            chat.pop(1)                  # system 바로 다음(가장 오래된) 턴 제거
            input_ids = encode(chat)

        # 텍스트 생성
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,             # *your code* — 생성 파라미터
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 생성된 부분만 디코딩 — 인스트럭트 모델은 답이 끝나면 스스로 멈춘다
        response = self.tokenizer.decode(
            output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        return response if response else "(응답을 생성하지 못했습니다)"