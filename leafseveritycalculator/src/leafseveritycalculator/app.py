import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW, BOTTOM, CENTER
import asyncio
import os
import time
import sys
import cv2
import numpy as np
from tatogalib.uri_io.urifilebrowser import UriFileBrowser
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

        self.main_window = toga.MainWindow(title="Leaf Severity Calculator")
        self.main_window.show()

        main_box = toga.Box(style=Pack(direction=COLUMN, padding=2, background_color='white', flex=1))
        container = toga.ScrollContainer(content=main_box)
        self.main_window.content = container

        buttons_box = toga.Box(style=Pack(direction=ROW, padding=2, background_color='white'))
        main_box.add(buttons_box)

        camera_button = toga.Button("Take a Photo", on_press=self.take_photo, style=Pack(padding=5, flex=1))
        buttons_box.add(camera_button)

        gallery_image_button = toga.Button("Select an Image", on_press=self.open_image, style=Pack(padding=5, flex=1))
        buttons_box.add(gallery_image_button)

        self.photo = toga.ImageView(style=Pack(height=300, padding=5, flex=1))
        main_box.add(self.photo)

        self.progress_label = toga.Label('', style=Pack(text_align='center'))
        main_box.add(self.progress_label)

        self.severity_button = toga.Button("Calculate Severity", on_press=self.process_image, style=Pack(padding=5), enabled=False)
        main_box.add(self.severity_button)

        self.result = toga.ImageView(style=Pack(height=300, padding=5, flex=1))
        main_box.add(self.result)

        self.lbl_severidad = toga.Label("", style=Pack(flex=1, font_size=18, font_weight='bold', text_align='center'))
        main_box.add(self.lbl_severidad)

        # Horizontal box of icons with white background
        iconos_box = toga.Box(style=Pack(direction=ROW, background_color='white', padding=10, alignment=BOTTOM, flex=1))

        icono_inicio = toga.Button(icon="resources/iconohome.png", on_press=self.go_home, 
                                   #style=Pack(padding_left=70, padding_right=20, background_color="white")
                                   style=Pack(padding_top=2, padding_bottom=0,flex=1, background_color="white", 
                                              alignment=CENTER)
                                   )
        iconos_box.add(icono_inicio)

        icono_guardar = toga.Button(icon="resources/iconoguardar.png", on_press=self.save_image, 
                                    #style=Pack(padding_left=20, padding_right=20, background_color="white")
                                    style=Pack(padding_top=2, padding_bottom=0,flex=1, background_color="white", 
                                                alignment=CENTER)
                                    )
        iconos_box.add(icono_guardar)
        
        icono_ayuda = toga.Button(icon="resources/iconoayuda.png", on_press=self.show_help,
                                  #style=Pack(padding_left=20, padding_right=20, background_color="white")
                                  style=Pack(padding_top=2, padding_bottom=0,flex=1, background_color="white", 
                                            alignment=CENTER)
                                  )
        iconos_box.add(icono_ayuda)

        icono_salir = toga.Button(icon="resources/iconosalir.png", on_press=self.exit_app, 
                                  #style=Pack(padding_left=20, padding_right=20, background_color="white")
                                  style=Pack(padding_top=2, padding_bottom=0,flex=1, background_color="white", 
                                             alignment=CENTER)
                                  )
        iconos_box.add(icono_salir)

        main_box.add(iconos_box)

        # Large institutional logos
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

    def go_home(self, widget):
        self.photo.image = None
        self.progress_label.text = ""
        self.severity_button.enabled = False
        self.result.image = None
        self.lbl_severidad.text = ""
    
    def show_help(self, widget):
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


    async def exit_app(self, widget):
        result = await self.main_window.confirm_dialog("Confirm Exit", "Do you want to close the application?")
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
                img_bytes = image.as_format(bytes)
                img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                self.img_original = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

                # Label indicating background correction
                self.progress_label.text = "Correcting illumination..."

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
                    _, png_buf = cv2.imencode('.png', cv2.cvtColor(image_corr, cv2.COLOR_RGB2BGR))
                    self.photo.image = toga.Image(png_buf.tobytes())
                    self.img_original = image_corr

                    # Label indicating background correction completed (show time if available)
                    if elapsed is not None:
                        self.progress_label.text = f"Illumination corrected ({elapsed:.1f}s)"
                    else:
                        self.progress_label.text = "Illumination corrected"
                    # Enable the process image button
                    self.severity_button.enabled = True 

                except Exception as e:
                    print(f"Error in background correction: {e}")
                    # Keep original capture usable even if correction fails.
                    self.progress_label.text = "Illumination correction failed. Using original image."
                    self.severity_button.enabled = True
                # Keep the original image if correction fails
                finally:
                    self.processing = False
                
                #image_corr = self.extract_background_color(np.array(image.as_format(Image.Image)))
                #self.photo.image = toga.Image(Image.fromarray(image_corr))
                #self.img_original = Image.fromarray(image_corr)

        except NotImplementedError:
            await self.main_window.dialog(toga.InfoDialog("Oh no!", "The Camera API is not implemented on this platform"))
        except PermissionError:
            await self.main_window.dialog(toga.InfoDialog("Oh no!", "You have not granted permission to take photos"))
        except Exception as e:
            await self.main_window.dialog(toga.InfoDialog("Error", f"Failed to capture/process photo: {str(e)}"))

    async def process_image(self, widget, **kwargs):
        if self.img_original is None:
            await self.main_window.dialog(toga.InfoDialog("Warning", "Please capture or select an image first."))
            return
        self.processing = True
        try:
            final_result = await asyncio.get_event_loop().run_in_executor(None, self._process_image_detailed)
            if final_result:
                processed_image, severity = final_result
                self.img_procesada = processed_image
                self.result.image = toga.Image(processed_image)
                self.severidad = severity
                self.lbl_severidad.text = f"Severity: {self.severidad:.2%}"
                self.severity_button.enabled = False
            else:
                await self.main_window.dialog(toga.InfoDialog("Error", "Image processing returned no result."))
        except Exception as e:
            await self.main_window.dialog(toga.InfoDialog("Error", f"Error processing image: {str(e)}"))
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
            img_np = np.array(self.img_original)
            img_resized = self._resize_preserve_aspect(img_np, 800, 600)

            r = img_resized[..., 0].astype(np.float32)
            g = img_resized[..., 1].astype(np.float32)
            b = img_resized[..., 2]

            epsilon = 1e-10
            indice = (g - r) / (g + r + epsilon)

            mascara_hojas = b <= self.ub_inicial
            mascara_enferma = np.logical_and(indice <= self.ui_inicial, mascara_hojas)
            mascara_sana = np.logical_and(indice > self.ui_inicial, mascara_hojas)

            severity = np.sum(mascara_enferma) / max(np.sum(mascara_hojas), 1)

            img_resultado = np.zeros_like(img_resized)
            img_resultado[mascara_sana] = [0, 255, 0]
            img_resultado[mascara_enferma] = [255, 0, 0]

            _, png_buf = cv2.imencode('.png', cv2.cvtColor(img_resultado, cv2.COLOR_RGB2BGR))
            return png_buf.tobytes(), severity
        except Exception as e:
            print(f"Error in OpenCV processing: {e}")
            return None

    def _resize_image(self, img, target_width=800, target_height=600):
        return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)

    def _resize_preserve_aspect(self, img, max_width=800, max_height=600):
        h, w = img.shape[:2]
        if w <= 0 or h <= 0:
            return img

        scale = min(max_width / w, max_height / h)
        scale = min(scale, 1.0)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _read_android_content_uri_bytes(self, uri_string):
        from android.net import Uri

        context = self._impl.native
        resolver = context.getContentResolver()
        stream = resolver.openInputStream(Uri.parse(uri_string))
        if stream is None:
            raise RuntimeError("Could not open Android content URI")

        try:
            data = bytearray()
            value = stream.read()
            while value != -1:
                data.append(value)
                value = stream.read()
            return bytes(data)
        finally:
            stream.close()

    async def open_image(self, widget, **kwargs):
        self.photo.image = None
        self.progress_label.text = ""
        self.severity_button.enabled = False
        self.result.image = None
        self.lbl_severidad.text = ""

        fb = UriFileBrowser()
        initial = "content://media/external/images/media"#"content://com.android.externalstorage.documents/document/camera"
        urilist = await fb.open_file_dialog("", file_types=["jpg"], initial_uri=initial, multiselect=False)

        if not urilist:
            return

        bytesobj = None
        try:
            from tatogalib.uri_io.urifile import UriFile
            urifile = UriFile(urilist[0])
            f = urifile.open("rb", "utf-8-sig", newline=None)
            try:
                bytesobj = f.read()
            finally:
                f.close()
        except Exception:
            if sys.platform == "android" and urilist[0].startswith("content://"):
                bytesobj = self._read_android_content_uri_bytes(urilist[0])
            else:
                raise
        

        self.photo.image = toga.Image(bytesobj)
        img_array = cv2.imdecode(np.frombuffer(bytesobj, np.uint8), cv2.IMREAD_COLOR)
        self.img_original = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        # Label indicating background correction
        self.progress_label.text = "Correcting illumination..."
        
        # Process background correction in the background
        self.processing = True
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                            None, self.extract_background_color, np.array(self.img_original)
                                    )

            if isinstance(result, tuple):
                image_corr, elapsed = result
            else:
                image_corr, elapsed = result, None

            # Update with corrected image
            _, png_buf = cv2.imencode('.png', cv2.cvtColor(image_corr, cv2.COLOR_RGB2BGR))
            self.photo.image = toga.Image(png_buf.tobytes())
            self.img_original = image_corr

            # Label indicating background correction completed
            if elapsed is not None:
                self.progress_label.text = f"Illumination corrected ({elapsed:.1f}s)"
            else:
                self.progress_label.text = "Illumination corrected"
            # Enable the process image button
            self.severity_button.enabled = True
                
        except Exception as e:
            print(f"Error in background correction: {e}")
            # Keep the original image if correction fails
            self.progress_label.text = "Illumination correction failed. Using selected image."
            self.severity_button.enabled = True
        finally:
            self.processing = False

        #image_corr = self.extract_background_color(np.array(toga.Image(bytesobj).as_format(Image.Image)))
        #self.photo.image = toga.Image(Image.fromarray(image_corr))
        #self.img_original = Image.fromarray(image_corr)

    def extract_background_color(self, image_rgb_original):
        RESIZE_FACTOR = 0.1
        ROLLING_RADIUS = 101
        t0 = time.time()

        image_rgb_small = cv2.resize(image_rgb_original, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR, interpolation=cv2.INTER_AREA)

        b, g, r = cv2.split(image_rgb_small)
        _, b_background = subtract_background_rolling_ball(b, ROLLING_RADIUS, light_background=True, use_paraboloid=False, do_presmooth=False)
        _, g_background = subtract_background_rolling_ball(g, ROLLING_RADIUS, light_background=True, use_paraboloid=False, do_presmooth=False)
        _, r_background = subtract_background_rolling_ball(r, ROLLING_RADIUS, light_background=True, use_paraboloid=False, do_presmooth=False)

        background_rgb_small = cv2.merge([b_background, g_background, r_background])
        background_rgb_full = cv2.resize(background_rgb_small, (image_rgb_original.shape[1], image_rgb_original.shape[0]), interpolation=cv2.INTER_CUBIC)
        image_corrected_rgb_full = cv2.subtract(background_rgb_full, image_rgb_original)
        corrected = cv2.bitwise_not(image_corrected_rgb_full)

        elapsed = time.time() - t0
        return corrected, elapsed

    async def save_image(self, widget, **kwargs):
        if self.img_procesada is None:
            await self.main_window.dialog(toga.InfoDialog("Warning", "No processed image to save."))
            return
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            safe_severity = f"{self.severidad:.2%}".replace("%", "pct")
            suggested_name = f"{timestamp}_Severity_{safe_severity}.png"

            fb = UriFileBrowser()
            save_uri = await fb.save_file_dialog(
                "Save segmented image",
                suggested_filename=suggested_name,
                file_types=["png"],
            )

            if not save_uri:
                return

            from tatogalib.uri_io.urifile import UriFile
            urifile = UriFile(save_uri)
            f = urifile.open("wb")
            try:
                f.write(self.img_procesada)
            finally:
                f.close()

            await self.main_window.dialog(toga.InfoDialog("Success", "Image saved successfully."))
        except Exception as e:
            await self.main_window.dialog(toga.InfoDialog("Error", f"Failed to save image: {str(e)}"))

def main():
    return LeafSeverityCalculator()
