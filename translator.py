from deep_translator import GoogleTranslator
from transformers import pipeline
import translators as ts
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import re
import requests
import urllib.parse


class MangaTranslator:
    def __init__(self):
        self.target = "en"
        self.target1 = "id"
        self.source = "ja"
        
        # Thread-safe model loading
        self._hf_ja_en_model = None
        self._hf_en_id_model = None
        self._hf_lock = threading.Lock()
        
        # Rate limiting untuk API calls
        self._last_api_call = {}
        self._api_lock = threading.Lock()
        
        self.translators = {
            "google": self._translate_with_google,
            "hf": self._translate_with_hf,
            "sogou": self._translate_with_sogou,
            "bing": self._translate_with_bing,
            "db": self._translate_with_db,
            "neko": self._translate_with_neko  # New Neko Labs translator
        }

    def translate(self, text, method="google"):
        """
        Translates the given text to the target language using the specified method.
        Thread-safe version with improved text preprocessing.
        """
        if not text or not text.strip():
            return text
            
        translator_func = self.translators.get(method)
        if translator_func:
            # Preprocess text before translation
            processed_text = self._preprocess_text(text)
            translated = translator_func(processed_text)
            # Post-process translated text
            return self._postprocess_text(translated)
        else:
            raise ValueError("Invalid translation method.")
    
    def translate_batch(self, texts, method="google"):
        """
        Translate multiple texts in parallel.
        
        Args:
            texts (list): List of texts to translate
            method (str): Translation method
            
        Returns:
            list: List of translated texts in same order
        """
        if not texts:
            return []
            
        # For offline HF model, we can process in true parallel
        if method == "hf":
            return self._translate_batch_hf(texts)
        elif method == "db":
            return self._translate_batch_db(texts)
        
        # For API-based methods, use controlled parallel processing
        return self._translate_batch_api(texts, method)
    
    def _translate_batch_hf(self, texts):
        """Parallel translation using HF model (offline) - JP to EN"""
        with self._hf_lock:
            if self._hf_ja_en_model is None:
                print("📄 Loading Helsinki-NLP/opus-mt-ja-en...")
                self._hf_ja_en_model = pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en")
        
        # Preprocess all texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # HF can handle batch processing natively
        try:
            results = self._hf_ja_en_model(processed_texts)
            translated_texts = [r["translation_text"] if r["translation_text"] else text 
                               for r, text in zip(results, processed_texts)]
            # Post-process all results
            return [self._postprocess_text(text) for text in translated_texts]
        except:
            # Fallback to sequential if batch fails
            return [self.translate(text, "hf") for text in texts]
    
    def _translate_batch_db(self, texts):
        """Double translation batch: JP->EN->ID using Helsinki models"""
        if not texts:
            return []
            
        # Load both models
        with self._hf_lock:
            if self._hf_ja_en_model is None:
                print("📄 Loading Helsinki-NLP/opus-mt-ja-en...")
                self._hf_ja_en_model = pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en")
            if self._hf_en_id_model is None:
                print("📄 Loading Helsinki-NLP/opus-mt-en-id...")
                self._hf_en_id_model = pipeline("translation", model="Helsinki-NLP/opus-mt-en-id")
        
        # Preprocess all texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        try:
            # First translation: JP -> EN
            en_results = self._hf_ja_en_model(processed_texts)
            en_texts = [r["translation_text"] if r["translation_text"] else text 
                       for r, text in zip(en_results, processed_texts)]
            
            # Second translation: EN -> ID
            id_results = self._hf_en_id_model(en_texts)
            final_texts = [r["translation_text"] if r["translation_text"] else en_text 
                          for r, en_text in zip(id_results, en_texts)]
            
            # Post-process all results
            return [self._postprocess_text(text) for text in final_texts]
        except Exception as e:
            print(f"⚠ Batch double translation failed: {e}")
            # Fallback to sequential
            return [self.translate(text, "db") for text in texts]
    
    def _translate_batch_api(self, texts, method):
        """Parallel translation using API methods with rate limiting"""
        results = [None] * len(texts)
        
        def translate_single(index, text):
            try:
                # Bypass the main translate method to avoid double preprocessing/postprocessing
                translator_func = self.translators.get(method)
                if translator_func:
                    processed_text = self._preprocess_text(text)
                    translated = translator_func(processed_text)
                    results[index] = self._postprocess_text(translated)
                else:
                    results[index] = text
            except:
                results[index] = text  # Fallback to original
        
        # Limit concurrent API calls to avoid rate limits
        max_workers = 2 if method in ["google", "bing", "sogou", "neko"] else 1
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(translate_single, i, text) 
                      for i, text in enumerate(texts)]
            
            # Wait for all to complete
            for future in futures:
                future.result()
        
        return results
            
    def _translate_with_google(self, text):
        self._rate_limit_api_call("google")
        try:
            translator = GoogleTranslator(source=self.source, target=self.target1)
            translated_text = translator.translate(text)
            return translated_text if translated_text is not None else text
        except Exception as e:
            print(f"Google translation error: {e}")
            return text

    def _translate_with_hf(self, text):
        """Helsinki JP->EN translation"""
        # Thread-safe HF model loading
        with self._hf_lock:
            if self._hf_ja_en_model is None:
                print("📄 Loading Helsinki-NLP/opus-mt-ja-en...")
                self._hf_ja_en_model = pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en")
            model = self._hf_ja_en_model
        
        try:
            translated_text = model(text)[0]["translation_text"]
            return translated_text if translated_text is not None else text
        except Exception as e:
            print(f"HF translation error: {e}")
            return text

    def _translate_with_sogou(self, text):
        self._rate_limit_api_call("sogou")
        try:
            translated_text = ts.translate_text(text, translator="sogou",
                                                from_language=self.source,
                                                to_language=self.target)
            return translated_text if translated_text is not None else text
        except Exception as e:
            print(f"Sogou translation error: {e}")
            return text

    def _translate_with_bing(self, text):
        self._rate_limit_api_call("bing")
        try:
            translated_text = ts.translate_text(text, translator="bing",
                                                from_language=self.source, 
                                                to_language=self.target1)
            return translated_text if translated_text is not None else text
        except Exception as e:
            print(f"Bing translation error: {e}")
            return text
    
    def _translate_with_neko(self, text):
        """Neko Labs API translation using Claude Sonnet-4"""
        self._rate_limit_api_call("neko")
        try:
            # URL encode the text
            encoded_text = urllib.parse.quote(text)
            system_prompt = urllib.parse.quote("kamu adalah ai yang straight forward, jadi gaperlu basa basi dan langsung kasih aku terjemahanya aja")
            
            # Build the API URL
            api_url = f"https://api.nekolabs.my.id/ai/claude/sonnet-4?text={encoded_text}&systemPrompt={system_prompt}"
            
            # Make the request with timeout
            response = requests.get(api_url, timeout=30)
            
            # Check if request was successful
            if response.status_code == 200:
                try:
                    json_response = response.json()
                    
                    # Check if response has the expected structure
                    if json_response.get("status") and json_response.get("result"):
                        translated_text = json_response["result"]
                        return translated_text if translated_text else text
                    else:
                        print(f"Neko API error: Invalid response structure")
                        return text
                        
                except ValueError as e:
                    print(f"Neko API error: Invalid JSON response - {e}")
                    return text
            else:
                print(f"Neko API error: HTTP {response.status_code}")
                return text
                
        except requests.exceptions.Timeout:
            print("Neko API error: Request timeout")
            return text
        except requests.exceptions.RequestException as e:
            print(f"Neko API error: {e}")
            return text
        except Exception as e:
            print(f"Neko API error: {e}")
            return text
    
    def _translate_with_db(self, text):
        """Double translation: JP->EN->ID using Helsinki models"""
        # Load both models thread-safely
        with self._hf_lock:
            if self._hf_ja_en_model is None:
                print("📄 Loading Helsinki-NLP/opus-mt-ja-en...")
                self._hf_ja_en_model = pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en")
            if self._hf_en_id_model is None:
                print("📄 Loading Helsinki-NLP/opus-mt-en-id...")
                self._hf_en_id_model = pipeline("translation", model="Helsinki-NLP/opus-mt-en-id")
            
            ja_en_model = self._hf_ja_en_model
            en_id_model = self._hf_en_id_model
            
        try:
            # First translation: JP -> EN
            en_result = ja_en_model(text)[0]["translation_text"]
            if not en_result:
                return text
                
            # Second translation: EN -> ID  
            id_result = en_id_model(en_result)[0]["translation_text"]
            return id_result if id_result else en_result
            
        except Exception as e:
            print(f"⚠ Double translation failed: {e}")
            return text

    def _preprocess_text(self, text):
        """
        Enhanced text preprocessing for better translation quality.
        """
        if not text:
            return text
            
        # Remove leading/trailing whitespace
        processed_text = text.strip()
        
        # Replace various Japanese punctuation with standard periods
        japanese_punctuation_map = {
            "．": ".",  # Full-width period
            "。": ".",  # Japanese period
            "！": "!",  # Full-width exclamation
            "？": "?",  # Full-width question mark
            "，": ",",  # Full-width comma
            "、": ",",  # Japanese comma
            "：": ":",  # Full-width colon
            "；": ";",  # Full-width semicolon
            "（": "(",  # Full-width left parenthesis
            "）": ")",  # Full-width right parenthesis
            # Removed quote replacements to preserve them
        }
        
        for jp_punct, en_punct in japanese_punctuation_map.items():
            processed_text = processed_text.replace(jp_punct, en_punct)
        
        # Remove extra whitespace and normalize spacing
        processed_text = re.sub(r'\s+', ' ', processed_text)
        
        # Handle common manga text patterns
        processed_text = processed_text.replace("．", ".")
        processed_text = processed_text.replace("...", ".")
        
        return processed_text.strip()

    def _postprocess_text(self, text):
        """
        Post-process translated text for better readability.
        """
        if not text:
            return text
            
        # Ensure consistent punctuation in output
        processed_text = text.strip()
        
        # Replace any remaining full-width punctuation
        processed_text = processed_text.replace("．", ".")
        processed_text = processed_text.replace("。", ".")
        processed_text = processed_text.replace("！", "!")
        processed_text = processed_text.replace("？", "?")
        
        # Clean up spacing around punctuation
        processed_text = re.sub(r'\s+([.!?,:;])', r'\1', processed_text)
        processed_text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', processed_text)
        
        # Remove multiple consecutive periods
        processed_text = re.sub(r'\.{2,}', '.', processed_text)
        
        # Capitalize first letter of sentences
        sentences = processed_text.split('. ')
        sentences = [s.capitalize() if s else s for s in sentences]
        processed_text = '. '.join(sentences)
        
        # Final cleanup
        processed_text = re.sub(r'\s+', ' ', processed_text)
        
        return processed_text.strip()

    def _rate_limit_api_call(self, api_name):
        """Thread-safe rate limiting for API calls"""
        with self._api_lock:
            current_time = time.time()
            last_call = self._last_api_call.get(api_name, 0)
            
            # Minimum delay between API calls
            min_delay = {
                "google": 1.0,
                "bing": 0.5,
                "sogou": 1.5,
                "neko": 2.0  # Neko API rate limit
            }.get(api_name, 1.0)
            
            time_since_last = current_time - last_call
            if time_since_last < min_delay:
                sleep_time = min_delay - time_since_last
                time.sleep(sleep_time)
            
            self._last_api_call[api_name] = time.time()

    def _delay(self):
        """Legacy delay method - now handled by rate limiting"""
        time.sleep(random.uniform(0.1, 0.5))  # Much shorter delay
        
    def get_model_info(self):
        """Get information about loaded models"""
        info = {
            "ja_en_loaded": self._hf_ja_en_model is not None,
            "en_id_loaded": self._hf_en_id_model is not None,
            "available_methods": list(self.translators.keys()),
            "preprocessing_enabled": True,
            "postprocessing_enabled": True
        }
        return info
    
    def test_translation(self, test_text="こんにちは", method="hf"):
        """Test translation functionality"""
        try:
            result = self.translate(test_text, method)
            return {
                "original": test_text,
                "translated": result,
                "method": method,
                "success": True
            }
        except Exception as e:
            return {
                "original": test_text,
                "translated": None,
                "method": method,
                "success": False,
                "error": str(e)
            }
