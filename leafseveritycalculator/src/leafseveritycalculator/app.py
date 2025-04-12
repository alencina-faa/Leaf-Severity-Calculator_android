"""
This app calculates the leaf severity from a photo or image from gallery - Optimized Version with Scientific Index
"""

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
import numpy as np
import asyncio
from PIL import Image
import io
import time
import cv2
from tatogalib.uri_io.urifilebrowser import UriFileBrowser
from tatogalib.uri_io.urifile import UriFile

class LeafSeverityCalculator(toga.App):
    def startup(self):
        self.img_original = None
        self.ui_inicial = -0.03365811811
        self.ub_inicial = 185
        self.severidad = 0
        self.processing = False
        self.cache = {}
        
        self.main_window = toga.MainWindow(title="Calculadora de Severidad de Hojas")
        self.main_window.show()

        main_box = toga.Box(style=Pack(direction=COLUMN, padding=2, background_color='#f0f0f0', flex=1))
        
        container = toga.ScrollContainer(content=main_box)
        self.main_window.content = container
        
        buttons_box = toga.Box(style=Pack(direction=ROW, padding=2, background_color='#f0f0f0'))
        main_box.add(buttons_box)

        camera_button = toga.Button(
            "Tomar una foto",
            on_press=self.take_photo,
            style=Pack(padding=5, flex=1)
        )
        buttons_box.add(camera_button)

        gallery_image_button = toga.Button(
            "Seleccionar una imagen",
            on_press=self.open_image,
            style=Pack(padding=5, flex=1)
        )
        buttons_box.add(gallery_image_button)

        self.photo = toga.ImageView(style=Pack(height=300, padding=5, flex=1))
        main_box.add(self.photo)


        severity_button = toga.Button(
            "Calcular la severidad",
            on_press=self.procesar_imagen,
            style=Pack(padding=5)
        )
        main_box.add(severity_button)

        self.result = toga.ImageView(style=Pack(height=300, padding=5, flex=1))
        main_box.add(self.result)

        self.lbl_severidad = toga.Label("", style=Pack(flex=1, font_size=18, font_weight='bold', text_align='center'))
        main_box.add(self.lbl_severidad)
        
        
    async def take_photo(self, widget, **kwargs):
        if self.processing:
            return
            
        try:
            if not self.camera.has_permission:
                await self.camera.request_permission()

            image = await self.camera.take_photo()

            if image:
                self.photo.image = image
                self.img_original = self.photo.image.as_format(Image.Image)

                
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
        
    async def procesar_imagen(self, widget, **kwargs):
        self.processing = True
        try:
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
            self.processing = False

    def _process_image_detailed(self):
        """Detailed image processing"""
        # Check cache first
        cache_key = hash(self.img_original.tobytes())
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = self._process_image_opencv()

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


    def _resize_image(self, img, max_dim):
        """Resize image keeping aspect ratio"""
            # OpenCV image
        height, width = img.shape[:2]
        scale = min(max_dim / width, max_dim / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    async def open_image(self, widget, **kwargs):
        """Opens an image from folder"""
        fb = UriFileBrowser()
        initial = "content://com.android.externalstorage.documents/document/camera"
        urilist = await fb.open_file_dialog("",
                        file_types=["jpg"],
                        initial_uri=initial,
                        multiselect= False
                        )
        
        urifile = UriFile(urilist[0])
        f = urifile.open("rb", "utf-8-sig", newline= None)
        bytesobj = f.read()
        f.close()
        self.photo.image = toga.Image(bytesobj)
        self.img_original = self.photo.image.as_format(Image.Image)



def main():
    return LeafSeverityCalculator()