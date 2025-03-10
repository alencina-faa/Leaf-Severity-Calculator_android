"""
This app calculates the leaf severity from a photo - Optimized Version with Scientific Index
"""

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
import numpy as np
import asyncio
from PIL import Image
import io
import time

try:
    import cv2
except ImportError:
    print("OpenCV not found. Please install it with: pip install opencv-python")
    cv2 = None

class LeafSeverityCalculator(toga.App):
    def startup(self):
        self.img_original = None
        self.ui_inicial = -0.03365811811
        self.ub_inicial = 185
        self.severidad = 0
        self.processing = False
        self.use_opencv = cv2 is not None
        self.cache = {}
        
        main_box = toga.Box(style=Pack(direction=COLUMN, padding=20, background_color='#f0f0f0'))

        title = toga.Label('Calculadora de Severidad de Hojas', style=Pack(text_align='center', font_size=24, font_weight='bold', padding=(0, 0, 20, 0)))
        main_box.add(title)

        self.photo = toga.ImageView(style=Pack(height=300, padding=5))
        camera_button = toga.Button(
            "Tomar foto",
            on_press=self.take_photo,
            style=Pack(padding=5)
        )

        self.lbl_severidad = toga.Label("", style=Pack(flex=1, font_size=18, font_weight='bold', text_align='center'))
        self.result = toga.ImageView(style=Pack(height=300, padding=5))
        severity_button = toga.Button(
            "Calcular la severidad",
            on_press=self.procesar_imagen,
            style=Pack(padding=5)
        )

        self.progress_label = toga.Label('', style=Pack(text_align='center'))
        
        engine_label = "Usando OpenCV" if self.use_opencv else "Usando PIL"
        self.engine_label = toga.Label(engine_label, style=Pack(text_align='center', padding=(0, 0, 10, 0)))

        main_box.add(self.engine_label)
        main_box.add(camera_button)
        main_box.add(self.photo)
        main_box.add(self.progress_label)
        main_box.add(severity_button)
        main_box.add(self.result)
        main_box.add(self.lbl_severidad)

        container = toga.ScrollContainer(content=main_box)

        self.main_window = toga.MainWindow(title="Calculadora de Severidad de Hojas")
        self.main_window.content = container
        self.main_window.show()

    async def take_photo(self, widget, **kwargs):
        if self.processing:
            return
            
        try:
            if not self.camera.has_permission:
                await self.camera.request_permission()

            self.progress_label.text = "Tomando foto..."
            image = await self.camera.take_photo()
            if image:
                self.photo.image = image
                self.img_original = self.photo.image.as_format(Image.Image)
                #await self.procesar_imagen()
        except NotImplementedError:
            await self.main_window.dialog(
                toga.InfoDialog(
                    "Oh no!",
                    "The Camera API is not implemented on this platform",
                )
            )
        except PermissionError:
            await self.main_window.dialog(
                toga.InfoDialog(
                    "Oh no!",
                    "You have not granted permission to take photos",
                )
            )
        finally:
            self.progress_label.text = ""

    async def procesar_imagen(self, widget, **kwargs):
        self.processing = True
        try:
            self.progress_label.text = "Procesando imagen..."
            
            # Two-stage processing
            # Stage 1: Quick preview
            preview_result = await asyncio.get_event_loop().run_in_executor(
                None, self._process_image_quick
            )
            
            if preview_result:
                preview_image, preview_severity = preview_result
                self.result.image = toga.Image(src=preview_image)
                self.lbl_severidad.text = f"Severidad (preliminar): {preview_severity:.2%}"
            
            # Stage 2: Detailed processing
            final_result = await asyncio.get_event_loop().run_in_executor(
                None, self._process_image_detailed
            )
            
            if final_result:
                processed_image, severity = final_result
                self.result.image = toga.Image(src=processed_image)
                self.severidad = severity
                self.lbl_severidad.text = f"Severidad: {self.severidad:.2%}"
                
        except Exception as e:
            await self.main_window.dialog(
                toga.InfoDialog(
                    "Error",
                    f"Error al procesar la imagen: {str(e)}"
                )
            )
        finally:
            self.progress_label.text = ""
            self.processing = False

    def _process_image_quick(self):
        """Quick preview processing"""
        try:
            # Use a very small image for quick processing
            img_small = self._resize_image(self.img_original, 100)
            
            # Simple thresholding for quick preview
            img_array = np.array(img_small)
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            
            # Use the original index, but with reduced precision for speed
            epsilon = 1e-7
            indice = (g.astype(np.float32) - r.astype(np.float32)) / (g.astype(np.float32) + r.astype(np.float32) + epsilon)
            
            mascara_hojas = b <= self.ub_inicial
            mascara_enferma = (indice <= self.ui_inicial) & mascara_hojas
            
            severity = np.sum(mascara_enferma) / np.sum(mascara_hojas)
            
            # Create a simple preview image
            preview = np.zeros_like(img_array)
            preview[mascara_hojas & ~mascara_enferma] = [0, 255, 0]
            preview[mascara_enferma] = [255, 0, 0]
            
            result_image = Image.fromarray(preview.astype(np.uint8))
            
            return result_image, severity
        except Exception as e:
            print(f"Error in quick processing: {e}")
            return None

    def _process_image_detailed(self):
        """Detailed image processing"""
        # Check cache first
        cache_key = hash(self.img_original.tobytes())
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if self.use_opencv:
            result = self._process_image_opencv()
        else:
            result = self._process_image_pil()
        
        # Cache the result
        self.cache[cache_key] = result
        return result

    def _process_image_opencv(self):
        try:
            img_np = np.array(self.img_original)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Resize for faster processing, max dimension 800
            img_resized = self._resize_image(img_cv, 800)
            
            # Split into BGR channels
            b, g, r = cv2.split(img_resized)
            
            # Convert to float32 for calculations
            r = r.astype(np.float32)
            g = g.astype(np.float32)
            
            # Calculate index with vectorized operations
            epsilon = 1e-10
            indice = cv2.divide(g - r, g + r + epsilon)
            
            # Create masks
            mascara_hojas = b <= self.ub_inicial
            mascara_enferma = np.logical_and(indice <= self.ui_inicial, mascara_hojas)
            mascara_sana = np.logical_and(indice > self.ui_inicial, mascara_hojas)
            
            # Calculate severity
            severity = np.sum(mascara_enferma) / max(np.sum(mascara_hojas), 1)
            
            # Create result image
            img_resultado = np.zeros_like(img_resized)
            img_resultado[mascara_sana] = [0, 255, 0]  # Green for healthy
            img_resultado[mascara_enferma] = [0, 0, 255]  # Red for diseased
            
            img_rgb = cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB)
            result_image = Image.fromarray(img_rgb)
            
            return result_image, severity
            
        except Exception as e:
            print(f"Error in OpenCV processing: {e}")
            return None

    def _process_image_pil(self):
        try:
            # Resize for faster processing, max dimension 800
            img_resized = self._resize_image(self.img_original, 800)
            
            img_array = np.array(img_resized)
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            
            # Calculate index
            epsilon = 1e-10
            indice = np.divide(g.astype(np.float32) - r.astype(np.float32), 
                               g.astype(np.float32) + r.astype(np.float32) + epsilon)
            
            mascara_hojas = b <= self.ub_inicial
            mascara_enferma = (indice <= self.ui_inicial) & mascara_hojas
            mascara_sana = (indice > self.ui_inicial) & mascara_hojas
            
            severity = np.sum(mascara_enferma) / max(np.sum(mascara_hojas), 1)
            
            img_resultado = np.zeros_like(img_array)
            img_resultado[mascara_sana] = [0, 255, 0]  # Green for healthy
            img_resultado[mascara_enferma] = [255, 0, 0]  # Red for diseased
            
            result_image = Image.fromarray(img_resultado.astype(np.uint8))
            
            return result_image, severity
            
        except Exception as e:
            print(f"Error in PIL processing: {e}")
            return None

    def _resize_image(self, img, max_dim):
        """Resize image keeping aspect ratio"""
        if isinstance(img, np.ndarray):
            # OpenCV image
            height, width = img.shape[:2]
            scale = min(max_dim / width, max_dim / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            # PIL image
            width, height = img.size
            scale = min(max_dim / width, max_dim / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

def main():
    return LeafSeverityCalculator()