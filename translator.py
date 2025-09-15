from deep_translator import GoogleTranslator
from transformers import pipeline
import translators as ts
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import queue


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
            "db": self._translate_with_db
        }

    def translate(self, text, method="google"):
        """
        Translates the given text to the target language using the specified method.
        Thread-safe version.
        """
        translator_func = self.translators.get(method)

        if translator_func:
            return translator_func(self._preprocess_text(text))
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
                print("🔄 Loading Helsinki-NLP/opus-mt-ja-en...")
                self._hf_ja_en_model = pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en")
        
        # HF can handle batch processing natively
        try:
            results = self._hf_ja_en_model(texts)
            return [r["translation_text"] if r["translation_text"] else text 
                   for r, text in zip(results, texts)]
        except:
            # Fallback to sequential if batch fails
            return [self._translate_with_hf(text) for text in texts]
    
    def _translate_batch_db(self, texts):
        """Double translation batch: JP->EN->ID using Helsinki models"""
        if not texts:
            return []
            
        # Load both models
        with self._hf_lock:
            if self._hf_ja_en_model is None:
                print("🔄 Loading Helsinki-NLP/opus-mt-ja-en...")
                self._hf_ja_en_model = pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en")
            if self._hf_en_id_model is None:
                print("🔄 Loading Helsinki-NLP/opus-mt-en-id...")
                self._hf_en_id_model = pipeline("translation", model="Helsinki-NLP/opus-mt-en-id")
        
        try:
            # First translation: JP -> EN
            en_results = self._hf_ja_en_model(texts)
            en_texts = [r["translation_text"] if r["translation_text"] else text 
                       for r, text in zip(en_results, texts)]
            
            # Second translation: EN -> ID
            id_results = self._hf_en_id_model(en_texts)
            final_texts = [r["translation_text"] if r["translation_text"] else en_text 
                          for r, en_text in zip(id_results, en_texts)]
            
            return final_texts
        except Exception as e:
            print(f"❌ Batch double translation failed: {e}")
            # Fallback to sequential
            return [self._translate_with_db(text) for text in texts]
    
    def _translate_batch_api(self, texts, method):
        """Parallel translation using API methods with rate limiting"""
        results = [None] * len(texts)
        
        def translate_single(index, text):
            try:
                results[index] = self.translate(text, method)
            except:
                results[index] = text  # Fallback to original
        
        # Limit concurrent API calls to avoid rate limits
        max_workers = 2 if method in ["google", "bing", "sogou"] else 1
        
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
        except:
            return text

    def _translate_with_hf(self, text):
        """Helsinki JP->EN translation"""
        # Thread-safe HF model loading
        with self._hf_lock:
            if self._hf_ja_en_model is None:
                print("🔄 Loading Helsinki-NLP/opus-mt-ja-en...")
                self._hf_ja_en_model = pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en")
            model = self._hf_ja_en_model
        
        try:
            translated_text = model(text)[0]["translation_text"]
            return translated_text if translated_text is not None else text
        except:
            return text

    def _translate_with_sogou(self, text):
        self._rate_limit_api_call("sogou")
        try:
            translated_text = ts.translate_text(text, translator="sogou",
                                                from_language=self.source,
                                                to_language=self.target)
            return translated_text if translated_text is not None else text
        except:
            return text

    def _translate_with_bing(self, text):
        self._rate_limit_api_call("bing")
        try:
            translated_text = ts.translate_text(text, translator="bing",
                                                from_language=self.source, 
                                                to_language=self.target1)
            return translated_text if translated_text is not None else text
        except:
            return text
    
    def _translate_with_db(self, text):
        """Double translation: JP->EN->ID using Helsinki models"""
        # Load both models thread-safely
        with self._hf_lock:
            if self._hf_ja_en_model is None:
                print("🔄 Loading Helsinki-NLP/opus-mt-ja-en...")
                self._hf_ja_en_model = pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en")
            if self._hf_en_id_model is None:
                print("🔄 Loading Helsinki-NLP/opus-mt-en-id...")
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
            print(f"❌ Double translation failed: {e}")
            return text

    def _preprocess_text(self, text):
        preprocessed_text = text.replace("ï¼Ž", ".")
        return preprocessed_text

    def _rate_limit_api_call(self, api_name):
        """Thread-safe rate limiting for API calls"""
        with self._api_lock:
            current_time = time.time()
            last_call = self._last_api_call.get(api_name, 0)
            
            # Minimum delay between API calls
            min_delay = {
                "google": 1.0,
                "bing": 0.5,
                "sogou": 1.5
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
            "available_methods": list(self.translators.keys())
        }

        return info
