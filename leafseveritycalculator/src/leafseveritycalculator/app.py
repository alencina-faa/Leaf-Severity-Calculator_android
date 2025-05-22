import toga
from toga.style import Pack
<<<<<<< Updated upstream
from toga.style.pack import COLUMN, ROW, BOTTOM, CENTER
=======
from toga.style.pack import COLUMN, ROW
import numpy as np
>>>>>>> Stashed changes
import asyncio
from PIL import Image
import io
import os
import time
import sys
import cv2
from tatogalib.uri_io.urifilebrowser import UriFileBrowser
from tatogalib.uri_io.urifile import UriFile
from cv2_rolling_ball import subtract_background_rolling_ball


class LeafSeverityCalculator(toga.App):
    def startup(self):
        self.img_original = None
        self.img_procesada = None
        self.ui_inicial = -0.03365811811
        self.ub_inicial = 185
        self.severidad = 0
        self.processing = False
        self.cache = {}
        
        self.main_window = toga.MainWindow(title="Calculadora de Severidad de Hojas")
        self.main_window.show()

<<<<<<< Updated upstream
        main_box = toga.Box(style=Pack(direction=COLUMN, padding=2, background_color='white', flex=1))
        container = toga.ScrollContainer(content=main_box)
        self.main_window.content = container

        buttons_box = toga.Box(style=Pack(direction=ROW, padding=2, background_color='white'))
=======
        main_box = toga.Box(style=Pack(direction=COLUMN, padding=2, background_color='#f0f0f0', flex=1))
        
        container = toga.ScrollContainer(content=main_box)
        self.main_window.content = container
        
        buttons_box = toga.Box(style=Pack(direction=ROW, padding=2, background_color='#f0f0f0'))
>>>>>>> Stashed changes
        main_box.add(buttons_box)

        camera_button = toga.Button("Tomar una foto", on_press=self.take_photo, style=Pack(padding=5, flex=1))
        buttons_box.add(camera_button)

        gallery_image_button = toga.Button("Seleccionar una imagen", on_press=self.open_image, style=Pack(padding=5, flex=1))
        buttons_box.add(gallery_image_button)

        self.photo = toga.ImageView(style=Pack(height=300, padding=5, flex=1))
        main_box.add(self.photo)

<<<<<<< Updated upstream
        severity_button = toga.Button("Calcular la severidad", on_press=self.procesar_imagen, style=Pack(padding=5))
=======

        severity_button = toga.Button(
            "Calcular la severidad",
            on_press=self.procesar_imagen,
            style=Pack(padding=5)
        )
>>>>>>> Stashed changes
        main_box.add(severity_button)

        self.result = toga.ImageView(style=Pack(height=300, padding=5, flex=1))
        main_box.add(self.result)

        self.lbl_severidad = toga.Label("", style=Pack(flex=1, font_size=18, font_weight='bold', text_align='center'))
        main_box.add(self.lbl_severidad)
<<<<<<< Updated upstream

        # Caja horizontal de íconos con fondo blanco
        iconos_box = toga.Box(style=Pack(direction=ROW, background_color='white', padding=10, alignment=CENTER, flex=1))

        icono_inicio = toga.Button(icon="resources/iconohome.png", on_press=self.inicio, style=Pack(padding_left=70, padding_right=20, background_color="white"))
        iconos_box.add(icono_inicio)

        icono_guardar = toga.Button(icon="resources/iconoguardar.png", on_press=self.guardar_imagen, style=Pack(padding_left=20, padding_right=20, background_color="white"))
        iconos_box.add(icono_guardar)
        
        icono_ayuda = toga.Button(
            icon="resources/iconoayuda.png",
            on_press=self.mostrar_ayuda,
            style=Pack(padding_left=20, padding_right=20, background_color="white")
        )
        iconos_box.add(icono_ayuda)

        icono_salir = toga.Button(icon="resources/iconosalir.png", on_press=self.salir, style=Pack(padding_left=20, padding_right=20, background_color="white"))
        iconos_box.add(icono_salir)

        main_box.add(iconos_box)

        # Logos institucionales grandes
        logos_row = toga.Box(style=Pack(direction=ROW, padding=10, alignment=CENTER))

        self.logo_uceva = toga.ImageView(
            toga.Image("resources/logo_uceva.png"),
            style=Pack(width=230, height=80, padding_left=10, padding_right=20, padding_bottom=70)
        )
        self.logo_faa = toga.ImageView(
            toga.Image("resources/LOGO_FAA.png"),
            style=Pack(width=230, height=70, padding=10,  padding_bottom=70)
        )
        logos_row.add(self.logo_uceva)
        logos_row.add(self.logo_faa)

        main_box.add(logos_row)

    def inicio(self, widget):
        self.photo.image = None
        self.result.image = None
        self.lbl_severidad.text = ""
    
    def mostrar_ayuda(self, widget):
        mensaje_corto = "This app calculates the leaf severity from a photo or an image."
        descripcion_larga = (
            "This application segments a photo or image of barley leaves pasted on "
           "a white sheet of paper into: background (black), healthy leaf portion (green), "
            "and diseased leaf portion (red). It then calculates the severity as the percentage "
           "of pixels in the diseased leaf regions relative to the total leaf pixels. "
            "The values used for segmentation were obtained from a sample of training images "
            "using the Otsu algorithm for the blue band and the Kmeans algorithm for the "
            "(red - green) / (red + green) index."
        )
        self.main_window.info_dialog("About This App", f"{mensaje_corto}\n\n{descripcion_larga}")


    async def salir(self, widget):
        result = await self.main_window.confirm_dialog("Confirmar salida", "¿Deseas cerrar la aplicación?")
        if result:
            import os
            import platform
            if platform.system() == "Java":
                from java.lang import System
                from android.os import Process
                Process.killProcess(Process.myPid())
            else:
                os._exit(0)

=======
        
        
>>>>>>> Stashed changes
    async def take_photo(self, widget, **kwargs):
        import numpy as np
        if self.processing:
            return
<<<<<<< Updated upstream
=======
            
>>>>>>> Stashed changes
        try:
            if not self.camera.has_permission:
                await self.camera.request_permission()
            image = await self.camera.take_photo()
            if image:
                image_corr = self.extract_background_color(np.array(image.as_format(Image.Image)))
                self.photo.image = toga.Image(Image.fromarray(image_corr))
<<<<<<< Updated upstream
                self.img_original = Image.fromarray(image_corr)
        except NotImplementedError:
            await self.main_window.dialog(toga.InfoDialog("Oh no!", "The Camera API is not implemented on this platform"))
        except PermissionError:
            await self.main_window.dialog(toga.InfoDialog("Oh no!", "You have not granted permission to take photos"))

    async def procesar_imagen(self, widget, **kwargs):
        self.processing = True
        try:
            final_result = await asyncio.get_event_loop().run_in_executor(None, self._process_image_detailed)
=======
                self.img_original = Image.fromarray(image_corr)#self.photo.image.as_format(Image.Image)

                
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
            
>>>>>>> Stashed changes
            if final_result:
                processed_image, severity = final_result
                self.img_procesada = processed_image
                self.result.image = toga.Image(src=processed_image)
                self.severidad = severity
                self.lbl_severidad.text = f"Severidad: {self.severidad:.2%}"
<<<<<<< Updated upstream
        except Exception as e:
            await self.main_window.dialog(toga.InfoDialog("Error", f"Error al procesar la imagen: {str(e)}"))
=======
                
        except Exception as e:
            await self.main_window.dialog(
                toga.InfoDialog(
                    "Error",
                    f"Error al procesar la imagen: {str(e)}"
                )
            )
>>>>>>> Stashed changes
        finally:
            self.processing = False

    def _process_image_detailed(self):
        """Detailed image processing"""
        # Check cache first
        cache_key = hash(self.img_original.tobytes())
        if cache_key in self.cache:
            return self.cache[cache_key]
<<<<<<< Updated upstream
=======
        
>>>>>>> Stashed changes
        result = self._process_image_opencv()

        # Cache the result
        self.cache[cache_key] = result
        return result

    def _process_image_opencv(self):
        import numpy as np
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
<<<<<<< Updated upstream
=======
            
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
        import numpy as np
        fb = UriFileBrowser()
        initial = "content://media/external/images/media"
        urilist = await fb.open_file_dialog("", file_types=["jpg"], initial_uri=initial, multiselect=False)

=======
        """Opens an image from folder"""
        fb = UriFileBrowser()
        initial = "content://com.android.externalstorage.documents/document/camera"
        urilist = await fb.open_file_dialog("",
                        file_types=["jpg"],
                        initial_uri=initial,
                        multiselect= False
                        )
        
>>>>>>> Stashed changes
        urifile = UriFile(urilist[0])
        f = urifile.open("rb", "utf-8-sig", newline= None)
        bytesobj = f.read()
        f.close()
        image_corr = self.extract_background_color(np.array(toga.Image(bytesobj).as_format(Image.Image)))
        self.photo.image = toga.Image(Image.fromarray(image_corr))
        self.img_original = Image.fromarray(image_corr)#self.photo.image.as_format(Image.Image)

    
    def extract_background_color(self, image_rgb_original):
        """Recibe una imagen, redimensiona, extrae fondo, redimensiona el fondo al tamaño original 
        y sustrae el fondo agrandado a la imagen original. El fondo final queda en blanco."""
        # Parámetros ajustables
        RESIZE_FACTOR = 0.1
        ROLLING_RADIUS = 101
<<<<<<< Updated upstream
        image_rgb_small = cv2.resize(image_rgb_original, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR, interpolation=cv2.INTER_AREA)

        b, g, r = cv2.split(image_rgb_small)
=======

        # Redimensionar la imagen original
        image_rgb_small = cv2.resize(image_rgb_original, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR, interpolation=cv2.INTER_AREA)
        
        # Separar canales de color
        b, g, r = cv2.split(image_rgb_small)
        # Aplicar Rolling Ball a cada canal
>>>>>>> Stashed changes
        _, b_background = subtract_background_rolling_ball(b, ROLLING_RADIUS, light_background=True, use_paraboloid=False, do_presmooth=True)
        _, g_background = subtract_background_rolling_ball(g, ROLLING_RADIUS, light_background=True, use_paraboloid=False, do_presmooth=True)
        _, r_background = subtract_background_rolling_ball(r, ROLLING_RADIUS, light_background=True, use_paraboloid=False, do_presmooth=True)
        
        # Imagen de fondo pequeña reconstruida
        background_rgb_small = cv2.merge([b_background, g_background, r_background])

        # Redimensionar el fondo al tamaño original
        background_rgb_full = cv2.resize(background_rgb_small, (image_rgb_original.shape[1], image_rgb_original.shape[0]), interpolation=cv2.INTER_CUBIC)

        # Sustraer el fondo redimensionado a la imagen original y crear fondo blanco
        image_corrected_rgb_full = cv2.subtract(background_rgb_full, image_rgb_original)
        return cv2.bitwise_not(image_corrected_rgb_full)

<<<<<<< Updated upstream
    async def guardar_imagen(self, widget, **kwargs):
        if self.img_procesada is None:
            await self.main_window.dialog(toga.InfoDialog("Advertencia", "No hay imagen procesada para guardar."))
            return
        try:
            save_dir = "/sdcard/Download/LeafSeverityImages"
            os.makedirs(save_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d")
            file_path = os.path.join(save_dir, f"{timestamp}_Severidad_{self.severidad:.2%}.png")
            output_bytes = io.BytesIO()
            self.img_procesada.save(output_bytes, format="PNG")
            with open(file_path, "wb") as f:
                f.write(output_bytes.getvalue())
            await self.main_window.dialog(toga.InfoDialog("Éxito", f"Imagen guardada en:\n{file_path}"))
        except Exception as e:
            await self.main_window.dialog(toga.InfoDialog("Error", f"No se pudo guardar la imagen: {str(e)}"))

def main():
    return LeafSeverityCalculator()
=======


def main():
    return LeafSeverityCalculator()
>>>>>>> Stashed changes
