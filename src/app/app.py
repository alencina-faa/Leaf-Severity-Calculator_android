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
        self.img_original = None
        self.mascara_hojas = None
        self.indice = None
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

        main_box.add(top_box)
        main_box.add(self.image_box)
        
        self.main_window = toga.MainWindow(title="Calculadora de Severidad de Hojas")
        self.main_window.content = main_box
        self.main_window.show()
    
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
                
                self.img_original = cv2.imread(str(file_path))
                if self.img_original is None:
                    raise ValueError(f"No se pudo cargar la imagen: {file_path}")
                                                               
                self.mostrar_imagen(cv2.resize(self.img_original, self.target_size), self.img_view_original)
                
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
            if self.img_original is not None and self.indice is not None:
                ub = self.ub_inicial
                ui = self.ui_inicial
                
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

def main():
    app = LeafSeverityCalculator('Calculadora de Severidad de Hojas', 'org.example.leafseveritycalculator')
    app.main_loop()

if __name__ == '__main__':
    main()