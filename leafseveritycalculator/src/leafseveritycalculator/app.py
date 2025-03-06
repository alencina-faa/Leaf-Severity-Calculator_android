"""
This app calculates the leaf severity from a photo - OpenCV Optimized Version
"""

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
import numpy as np
import asyncio
import io
from PIL import Image  # Still needed for Toga compatibility

# Import OpenCV
try:
    import cv2
except ImportError:
    print("OpenCV not found. Please install it with: pip install opencv-python")
    # Fallback to PIL if OpenCV is not available
    cv2 = None


class LeafSeverityCalculator(toga.App):
    def startup(self):
        self.img_original = None
        self.ui_inicial = -0.03365811811
        self.ub_inicial = 185
        self.severidad = 0
        self.processing = False
        self.use_opencv = cv2 is not None
        
        # Main box
        main_box = toga.Box(style=Pack(direction=COLUMN, padding=20, background_color='#f0f0f0'))

        # Title
        title = toga.Label('Calculadora de Severidad de Hojas', style=Pack(text_align='center', font_size=24, font_weight='bold', padding=(0, 0, 20, 0)))
        main_box.add(title)

        self.photo = toga.ImageView(style=Pack(height=300, padding=5))
        camera_button = toga.Button(
            "Tomar foto",
            on_press=self.take_photo,
            style=Pack(padding=5)
        )

        self.lbl_severidad = toga.Label('Severidad: ', style=Pack(flex=1, font_size=18, font_weight='bold', text_align='center'))
        self.result = toga.ImageView(style=Pack(height=300, padding=5))
        self.progress_label = toga.Label('', style=Pack(text_align='center'))
        
        # Add a label to show if using OpenCV or PIL
        engine_label = "Usando OpenCV" if self.use_opencv else "Usando PIL"
        self.engine_label = toga.Label(engine_label, style=Pack(text_align='center', padding=(0, 0, 10, 0)))

        main_box.add(self.engine_label)
        main_box.add(camera_button)
        main_box.add(self.photo)
        main_box.add(self.progress_label)
        main_box.add(self.lbl_severidad)
        main_box.add(self.result)

        self.main_window = toga.MainWindow(title="Calculadora de Severidad de Hojas")
        self.main_window.content = main_box
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
                await self.procesar_imagen()
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

    async def procesar_imagen(self):
        self.processing = True
        try:
            self.progress_label.text = "Procesando imagen..."
            
            # Process in a separate thread to avoid UI freezing
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._process_image_opencv if self.use_opencv else self._process_image_pil
            )
            
            if result:
                processed_image, severity = result
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

    def _process_image_opencv(self):
        """Process image using OpenCV for better performance"""
        try:
            # Convert PIL Image to OpenCV format
            img_np = np.array(self.img_original)
            # Convert RGB to BGR (OpenCV format)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Resize for faster processing (maintain aspect ratio)
            height, width = img_cv.shape[:2]
            max_dim = 800
            scale = max_dim / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_resized = cv2.resize(img_cv, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Split into BGR channels (OpenCV uses BGR)
            b, g, r = cv2.split(img_resized)
            
            # Convert to float32 for calculations
            r = r.astype(np.float32)
            g = g.astype(np.float32)
            
            # Calculate index with vectorized operations
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            indice = cv2.divide(g - r, g + r + epsilon)
            
            # Create masks
            mascara_hojas = b <= self.ub_inicial
            mascara_enferma = np.logical_and(indice <= self.ui_inicial, mascara_hojas)
            mascara_sana = np.logical_and(indice > self.ui_inicial, mascara_hojas)
            
            # Create result image (start with zeros)
            img_resultado = np.zeros_like(img_resized)
            
            # Apply masks - OpenCV uses BGR
            img_resultado[mascara_sana] = [0, 255, 0]  # Green for healthy
            img_resultado[mascara_enferma] = [0, 0, 255]  # Red for diseased (BGR)
            
            # Calculate severity
            num_enferma = np.count_nonzero(mascara_enferma)
            num_hojas = np.count_nonzero(mascara_hojas)
            severity = num_enferma / max(num_hojas, 1)
            
            # Convert back to RGB for PIL
            img_rgb = cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image for Toga
            result_image = Image.fromarray(img_rgb)
            
            return result_image, severity
            
        except Exception as e:
            print(f"Error in OpenCV processing: {e}")
            return None

    def _process_image_pil(self):
        """Fallback to PIL processing if OpenCV is not available"""
        try:
            # Resize image for faster processing
            width, height = self.img_original.size
            max_dim = 800
            scale = max_dim / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_resized = self.img_original.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to numpy array once
            img_array = np.array(img_resized)
            
            # Extract channels directly from the array
            r = img_array[:, :, 0].astype(np.float32)
            g = img_array[:, :, 1].astype(np.float32)
            b = img_array[:, :, 2].astype(np.uint8)
            
            # Calculate index with vectorized operations
            epsilon = 1e-10
            indice = np.divide(g - r, g + r + epsilon)
            
            # Create masks with vectorized operations
            mascara_hojas = b <= self.ub_inicial
            mascara_enferma = (indice <= self.ui_inicial) & mascara_hojas
            mascara_sana = (indice > self.ui_inicial) & mascara_hojas
            
            # Create result image efficiently
            img_resultado = np.zeros_like(img_array)
            
            # Apply masks in a vectorized way
            img_resultado[mascara_sana] = [0, 255, 0]  # Green for healthy
            img_resultado[mascara_enferma] = [255, 0, 0]  # Red for diseased
            
            # Calculate severity
            severity = np.sum(mascara_enferma) / max(np.sum(mascara_hojas), 1)
            
            # Convert back to PIL
            result_image = Image.fromarray(img_resultado.astype(np.uint8))
            
            return result_image, severity
            
        except Exception as e:
            print(f"Error in PIL processing: {e}")
            return None


def main():
    return LeafSeverityCalculator()