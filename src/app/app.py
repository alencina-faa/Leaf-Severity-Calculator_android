"""
My first application
"""

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
import cv2
import numpy as np
from PIL import Image
import io
from pathlib import Path

class LeafSeverityCalculator(toga.App):
    def startup(self):
        self.img = None
        self.img_original = None
        self.mascara_hojas = None
        self.indice = None
        self.nombre_archivo = ""
        self.ui_inicial = -0.030792934
        self.ub_inicial = 180
        self.severidad = 0
        self.target_size = (400, 400)

        # Main box
        main_box = toga.Box(style=Pack(direction=COLUMN, padding=20, background_color='#f0f0f0'))

        # Title
        title = toga.Label('Calculadora de Severidad de Hojas', style=Pack(text_align='center', font_size=24, font_weight='bold', padding=(0, 0, 20, 0)))
        main_box.add(title)

        # Top box
        top_box = toga.Box(style=Pack(direction=ROW, padding=(0, 0, 20, 0)))
        self.btn_cargar = toga.Button('Cargar Imagen', on_press=self.cargar_imagen, style=Pack(flex=1, padding=5))
        self.lbl_severidad = toga.Label('Severidad: ', style=Pack(flex=1, font_size=18, font_weight='bold', text_align='right'))
        top_box.add(self.btn_cargar)
        top_box.add(self.lbl_severidad)

        # Image box
        self.image_box = toga.Box(style=Pack(direction=ROW, padding=(0, 10)))
        self.img_view_original = toga.ImageView(style=Pack(flex=1, width=400, height=400))
        self.img_view_procesada = toga.ImageView(style=Pack(flex=1, width=400, height=400))
        self.image_box.add(self.img_view_original)
        self.image_box.add(self.img_view_procesada)

        # Control box
        control_box = toga.Box(style=Pack(direction=COLUMN, padding=(20, 0)))
        self.slider_background, self.entry_fondo = self.crear_control_deslizador("Umbral de Fondo", 0, 255, self.actualizar_umbral_b, self.ub_inicial)
        self.slider_disease, self.entry_enfermedad = self.crear_control_deslizador("Umbral de Enfermedad", -100, 100, self.actualizar_umbral_indice, int(self.ui_inicial * 100))
        control_box.add(self.slider_background)
        control_box.add(self.slider_disease)

        main_box.add(top_box)
        main_box.add(self.image_box)
        main_box.add(control_box)

        self.main_window = toga.MainWindow(title="Calculadora de Severidad de Hojas")
        self.main_window.content = main_box
        self.main_window.show()

    def crear_control_deslizador(self, label, min_val, max_val, on_change, valor_inicial):
        box = toga.Box(style=Pack(direction=COLUMN, padding=(0, 10)))
        lbl = toga.Label(label, style=Pack(padding=(0, 0, 5, 0)))
        slider = toga.Slider(
            range=(min_val, max_val),
            value=valor_inicial,
            on_change=on_change,
            style=Pack(width=400, padding=(0, 5))  # Aumentamos el ancho del slider
        )
        entry = toga.TextInput(value=str(valor_inicial), style=Pack(width=100))
        entry.on_change = lambda widget: self.actualizar_slider_desde_entrada(slider, entry, min_val, max_val)
        
        slider_box = toga.Box(style=Pack(direction=ROW, alignment='center'))
        slider_box.add(slider)
        slider_box.add(entry)
        
        box.add(lbl)
        box.add(slider_box)
        
        return box, entry

    def actualizar_slider_desde_entrada(self, slider, entry, min_val, max_val):
        try:
            value = float(entry.value)
            if min_val <= value <= max_val:
                slider.value = value
            else:
                entry.value = str(slider.value)
        except ValueError:
            entry.value = str(slider.value)

    async def cargar_imagen(self, widget):
        try:
            file_dialog_result = await self.main_window.open_file_dialog(
                title="Seleccionar una imagen",
                multiselect=False,
                file_types=['jpg', 'png']
            )
            if file_dialog_result:
                file_path = file_dialog_result[0] if isinstance(file_dialog_result, (list, tuple)) else file_dialog_result
                file_path = Path(file_path)
                
                self.nombre_archivo = file_path.name
                self.img_original = cv2.imread(str(file_path))
                if self.img_original is None:
                    raise ValueError(f"No se pudo cargar la imagen: {file_path}")
                
                self.img_original = cv2.resize(self.img_original, self.target_size)
                self.img = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2RGB)
                
                self.mostrar_imagen(self.img_original, self.img_view_original)
                
                self.calcular_indice()
                
                self.procesar_imagen()
            else:
                print("No se seleccionó ningún archivo")
        except Exception as e:
            print(f"Error al cargar la imagen: {e}")
            await self.main_window.info_dialog('Error', str(e))

    def mostrar_imagen(self, img, img_view):
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)
            
            with io.BytesIO() as buffer:
                pil_image.save(buffer, format="PNG")
                img_data = buffer.getvalue()
            
            img_view.image = toga.Image(data=img_data)
        except Exception as e:
            print(f"Error al mostrar la imagen: {e}")
            self.main_window.info_dialog('Error al mostrar imagen', str(e))

    def calcular_indice(self):
        try:
            _, g, r = cv2.split(self.img_original)
            self.indice = np.divide(
                g.astype(float) - r.astype(float), 
                g.astype(float) + r.astype(float), 
                out=np.zeros_like(g, dtype=float), 
                where=(g+r) != 0
            )
        except Exception as e:
            print(f"Error al calcular el índice de enfermedad: {e}")
            self.main_window.info_dialog('Error al calcular índice', str(e))
    
    def procesar_imagen(self):
        try:
            if self.img is not None and self.indice is not None:
                ub = self.slider_background.children[1].children[0].value
                ui = self.slider_disease.children[1].children[0].value / 100
                
                b, _, _ = cv2.split(self.img_original)
                self.mascara_hojas = b <= ub
                
                mascara_enferma = (self.indice <= ui) & self.mascara_hojas
                mascara_sana = (self.indice > ui) & self.mascara_hojas
                
                img_resultado = self.img_original.copy()
                img_resultado[~self.mascara_hojas] = [0, 0, 0]
                img_resultado[mascara_enferma] = [0, 0, 255]
                img_resultado[mascara_sana] = [0, 255, 0]
                
                self.severidad = np.sum(mascara_enferma) / np.sum(self.mascara_hojas)
                self.lbl_severidad.text = f"Severidad: {self.severidad:.2%}"
                
                self.mostrar_imagen(img_resultado, self.img_view_procesada)
        except Exception as e:
            print(f"Error al procesar la imagen: {e}")
            self.main_window.info_dialog('Error al procesar imagen', str(e))

    def actualizar_umbral_b(self, widget):
        self.entry_fondo.value = str(int(widget.value))
        self.procesar_imagen()

    def actualizar_umbral_indice(self, widget):
        self.entry_enfermedad.value = f"{widget.value / 100:.2f}"
        self.procesar_imagen()
def main():
    app = LeafSeverityCalculator('Calculadora de Severidad de Hojas', 'org.example.leafseveritycalculator')
    app.main_loop()

if __name__ == '__main__':
    main()

