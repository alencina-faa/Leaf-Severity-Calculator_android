import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW, BOTTOM, CENTER
import asyncio
from PIL import Image
import io
import os
import time
import sys
import numpy as np
from tatogalib.uri_io.urifilebrowser import UriFileBrowser
from tatogalib.uri_io.urifile import UriFile
from numpy_rolling_ball import subtract_background_rolling_ball


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

        main_box = toga.Box(style=Pack(direction=COLUMN, padding=2, background_color='white', flex=1))
        container = toga.ScrollContainer(content=main_box)
        self.main_window.content = container

        buttons_box = toga.Box(style=Pack(direction=ROW, padding=2, background_color='white'))
        main_box.add(buttons_box)

        camera_button = toga.Button("Tomar una foto", on_press=self.take_photo, style=Pack(padding=5, flex=1))
        buttons_box.add(camera_button)

        gallery_image_button = toga.Button("Seleccionar una imagen", on_press=self.open_image, style=Pack(padding=5, flex=1))
        buttons_box.add(gallery_image_button)

        self.photo = toga.ImageView(style=Pack(height=300, padding=5, flex=1))
        main_box.add(self.photo)

        self.progress_label = toga.Label('', style=Pack(text_align='center'))
        main_box.add(self.progress_label)

        self.severity_button = toga.Button("Calcular la severidad", on_press=self.procesar_imagen, style=Pack(padding=5), enabled=False)
        main_box.add(self.severity_button)

        self.result = toga.ImageView(style=Pack(height=300, padding=5, flex=1))
        main_box.add(self.result)

        self.lbl_severidad = toga.Label("", style=Pack(flex=1, font_size=18, font_weight='bold', text_align='center'))
        main_box.add(self.lbl_severidad)

        # Caja horizontal de íconos con fondo blanco
        iconos_box = toga.Box(style=Pack(direction=ROW, background_color='white', padding=10, alignment=BOTTOM, flex=1))

        icono_inicio = toga.Button(icon="resources/iconohome.png", on_press=self.inicio, 
                                   #style=Pack(padding_left=70, padding_right=20, background_color="white")
                                   style=Pack(padding_top=2, padding_bottom=0,flex=1, background_color="white", 
                                              alignment=CENTER)
                                   )
        iconos_box.add(icono_inicio)

        icono_guardar = toga.Button(icon="resources/iconoguardar.png", on_press=self.guardar_imagen, 
                                    #style=Pack(padding_left=20, padding_right=20, background_color="white")
                                    style=Pack(padding_top=2, padding_bottom=0,flex=1, background_color="white", 
                                                alignment=CENTER)
                                    )
        iconos_box.add(icono_guardar)
        
        icono_ayuda = toga.Button(icon="resources/iconoayuda.png", on_press=self.mostrar_ayuda,
                                  #style=Pack(padding_left=20, padding_right=20, background_color="white")
                                  style=Pack(padding_top=2, padding_bottom=0,flex=1, background_color="white", 
                                            alignment=CENTER)
                                  )
        iconos_box.add(icono_ayuda)

        icono_salir = toga.Button(icon="resources/iconosalir.png", on_press=self.salir, 
                                  #style=Pack(padding_left=20, padding_right=20, background_color="white")
                                  style=Pack(padding_top=2, padding_bottom=0,flex=1, background_color="white", 
                                             alignment=CENTER)
                                  )
        iconos_box.add(icono_salir)

        main_box.add(iconos_box)

        # Logos institucionales grandes
        logos_row = toga.Box(style=Pack(direction=ROW, padding=5, background_color='#f0f0f0', 
                                        flex=1, alignment=BOTTOM, height=70))

        self.logo_uceva = toga.ImageView(
            toga.Image("resources/logo_uceva.png"),
            style=Pack(padding_top=2, padding_bottom=5,flex=1, alignment=CENTER)
        )
        self.logo_faa = toga.ImageView(
            toga.Image("resources/LOGO_FAA.png"),
            style=Pack(padding_top=2, padding_bottom=5,flex=1, alignment=CENTER)
        )
        logos_row.add(self.logo_uceva)
        logos_row.add(self.logo_faa)

        main_box.add(logos_row)

        logo_cic_container = toga.Box(style=Pack(direction=ROW, padding=0, background_color="#00aec3",
                                                flex=1, alignment=BOTTOM, height=70))
        logo_cic = toga.ImageView(
            toga.Image("resources/logoCIC.png"),
            style=Pack(padding_top=2, padding_bottom=5,flex=1, alignment=CENTER)
        )
        logo_cic_container.add(logo_cic)
        main_box.add(logo_cic_container)

    def inicio(self, widget):
        self.photo.image = None
        self.progress_label.text = ""
        self.severity_button.enabled = False
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

    async def take_photo(self, widget, **kwargs):
        import numpy as np
        self.photo.image = None
        self.progress_label.text = ""
        self.severity_button.enabled = False
        self.result.image = None
        self.lbl_severidad.text = ""

        if self.processing:
            return
        try:
            if not self.camera.has_permission:
                await self.camera.request_permission()
            image = await self.camera.take_photo()
            if image:
                self.photo.image = image
                self.img_original = image.as_format(Image.Image)

                # Label indicando la corrección de fondo
                self.progress_label.text = "Corrigiendo iluminación..."

                # Procesar la corrección de fondo en segundo plano
                self.processing = True
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                                        None, self.extract_background_color, np.array(self.img_original)
                                    )
                
                    # extract_background_color now returns (image_corr, elapsed)
                    if isinstance(result, tuple):
                        image_corr, elapsed = result
                    else:
                        image_corr, elapsed = result, None

                    # Actualizar con la imagen corregida
                    self.photo.image = toga.Image(Image.fromarray(image_corr))
                    self.img_original = Image.fromarray(image_corr)

                    # Label indicando corrección de fondo concluída (mostrar tiempo si disponible)
                    if elapsed is not None:
                        self.progress_label.text = f"Iluminación corregida ({elapsed:.1f}s)"
                    else:
                        self.progress_label.text = "Iluminación corregida"
                    # Habilitar el botón de procesar imagen
                    self.severity_button.enabled = True 

                except Exception as e:
                    print(f"Error en corrección de fondo: {e}")
                # Mantener la imagen original si falla la corrección
                finally:
                    self.processing = False
                
                #image_corr = self.extract_background_color(np.array(image.as_format(Image.Image)))
                #self.photo.image = toga.Image(Image.fromarray(image_corr))
                #self.img_original = Image.fromarray(image_corr)

        except NotImplementedError:
            await self.main_window.dialog(toga.InfoDialog("Oh no!", "The Camera API is not implemented on this platform"))
        except PermissionError:
            await self.main_window.dialog(toga.InfoDialog("Oh no!", "You have not granted permission to take photos"))

    async def procesar_imagen(self, widget, **kwargs):
        self.processing = True
        try:
            final_result = await asyncio.get_event_loop().run_in_executor(None, self._process_image_detailed)
            if final_result:
                processed_image, severity = final_result
                self.img_procesada = processed_image
                self.result.image = toga.Image(src=processed_image)
                self.severidad = severity
                self.lbl_severidad.text = f"Severidad: {self.severidad:.2%}"
                self.severity_button.enabled = False
        except Exception as e:
            await self.main_window.dialog(toga.InfoDialog("Error", f"Error al procesar la imagen: {str(e)}"))
        finally:
            self.processing = False

    def _process_image_detailed(self):
        cache_key = hash(self.img_original.tobytes())
        if cache_key in self.cache:
            return self.cache[cache_key]
        result = self._process_image_opencv()
        self.cache[cache_key] = result
        return result

    def _process_image_opencv(self):
        try:
            # Ensure we have a NumPy array in RGB order
            img_np = np.array(self.img_original)

            # Resize image to the common processing target size.
            img_resized = self._resize_image_numpy(img_np, 800, 600)

            # Extract channels (assume RGB input)
            r = img_resized[..., 0].astype(np.float32)
            g = img_resized[..., 1].astype(np.float32)
            b = img_resized[..., 2]

            epsilon = 1e-10
            indice = (g - r) / (g + r + epsilon)

            mascara_hojas = b <= self.ub_inicial
            mascara_enferma = np.logical_and(indice <= self.ui_inicial, mascara_hojas)
            mascara_sana = np.logical_and(indice > self.ui_inicial, mascara_hojas)

            severity = np.sum(mascara_enferma) / max(np.sum(mascara_hojas), 1)

            # Prepare output image (RGB)
            img_resultado = np.zeros_like(img_resized)
            img_resultado[mascara_sana] = [0, 255, 0]
            img_resultado[mascara_enferma] = [255, 0, 0]  # red in RGB

            result_image = Image.fromarray(img_resultado.astype('uint8'))
            return result_image, severity
        except Exception as e:
            print(f"Error in NumPy processing: {e}")
            return None

    def _resize_image(self, img, target_width=800, target_height=600):
        # kept for backward compatibility but delegate to numpy implementation
        return self._resize_image_numpy(img, target_width, target_height)

    def _resize_image_numpy(self, img, target_width=800, target_height=600):
        return self._resize_with_pillow(img, target_width, target_height)

    def _resize_with_pillow(self, img, new_w, new_h):
        """Resize a numpy array or PIL Image using Pillow Lanczos resampling and return a numpy array.

        new_w, new_h are integers (width, height).
        """
        # If input is a numpy array, convert to PIL Image first
        if not isinstance(img, Image.Image):
            pil = Image.fromarray(img)
        else:
            pil = img

        # Pillow expects size as (width, height)
        resized = pil.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
        arr = np.array(resized)
        # Ensure uint8 output
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr

    async def open_image(self, widget, **kwargs):
        import numpy as np
        self.photo.image = None
        self.progress_label.text = ""
        self.severity_button.enabled = False
        self.result.image = None
        self.lbl_severidad.text = ""

        fb = UriFileBrowser()
        initial = "content://media/external/images/media"#"content://com.android.externalstorage.documents/document/camera"
        urilist = await fb.open_file_dialog("", file_types=["jpg"], initial_uri=initial, multiselect=False)

        urifile = UriFile(urilist[0])
        f = urifile.open("rb", "utf-8-sig", newline=None)
        bytesobj = f.read()
        f.close()
        

        self.photo.image = toga.Image(bytesobj)
        self.img_original = toga.Image(bytesobj).as_format(Image.Image)

        # Label indicando la corrección de fondo
        self.progress_label.text = "Corrigiendo iluminación..."
        
        # Procesar la corrección de fondo en segundo plano
        self.processing = True
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                            None, self.extract_background_color, np.array(self.img_original)
                                    )
                
            if isinstance(result, tuple):
                image_corr, elapsed = result
            else:
                image_corr, elapsed = result, None

            # Actualizar con la imagen corregida
            self.photo.image = toga.Image(Image.fromarray(image_corr))
            self.img_original = Image.fromarray(image_corr)

            # Label indicando corrección de fondo concluída
            if elapsed is not None:
                self.progress_label.text = f"Iluminación corregida ({elapsed:.1f}s)"
            else:
                self.progress_label.text = "Iluminación corregida"
            # Habilitar el botón de procesar imagen
            self.severity_button.enabled = True
                
        except Exception as e:
            print(f"Error en corrección de fondo: {e}")
            # Mantener la imagen original si falla la corrección
        finally:
            self.processing = False

        #image_corr = self.extract_background_color(np.array(toga.Image(bytesobj).as_format(Image.Image)))
        #self.photo.image = toga.Image(Image.fromarray(image_corr))
        #self.img_original = Image.fromarray(image_corr)

    def extract_background_color(self, image_rgb_original):
        RESIZE_FACTOR = 0.1
        ROLLING_RADIUS = 101
        # measure elapsed time for diagnostics
        t0 = time.time()

        # Resize using our numpy resize (via Pillow)
        small_w = max(int(image_rgb_original.shape[1] * RESIZE_FACTOR), 1)
        small_h = max(int(image_rgb_original.shape[0] * RESIZE_FACTOR), 1)
        image_rgb_small = self._resize_with_pillow(image_rgb_original, small_w, small_h)

        # split channels (assume HxWx3)
        b = image_rgb_small[..., 2] if image_rgb_small.ndim == 3 else image_rgb_small
        g = image_rgb_small[..., 1] if image_rgb_small.ndim == 3 else image_rgb_small
        r = image_rgb_small[..., 0] if image_rgb_small.ndim == 3 else image_rgb_small

        # Disable presmooth due to an issue in the smoothing routine of numpy_rolling_ball
        # (presmooth changes array shape in some versions). This also improves speed.
        _, b_background = subtract_background_rolling_ball(b, ROLLING_RADIUS, light_background=True, use_paraboloid=False, do_presmooth=False)
        _, g_background = subtract_background_rolling_ball(g, ROLLING_RADIUS, light_background=True, use_paraboloid=False, do_presmooth=False)
        _, r_background = subtract_background_rolling_ball(r, ROLLING_RADIUS, light_background=True, use_paraboloid=False, do_presmooth=False)

        # merge backgrounds back to small rgb (we'll keep RGB order)
        background_rgb_small = np.stack([r_background, g_background, b_background], axis=-1)

        # resize background to full size and subtract
        background_rgb_full = self._resize_with_pillow(background_rgb_small, image_rgb_original.shape[1], image_rgb_original.shape[0])
        image_corrected_rgb_full = background_rgb_full.astype(np.int16) - image_rgb_original.astype(np.int16)
        image_corrected_rgb_full = np.clip(image_corrected_rgb_full, 0, 255).astype(np.uint8)
        # invert (bitwise_not equivalent)
        corrected = 255 - image_corrected_rgb_full

        elapsed = time.time() - t0
        return corrected, elapsed

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
