from app import LeafSeverityCalculator 

def main():
    # Crear la instancia de la aplicación
    app = LeafSeverityCalculator('Calculadora de Severidad de Hojas', 'org.example.leafseveritycalculator')
    # Iniciar el ciclo de eventos de la aplicación
    app.main_loop()

if __name__ == '__main__':
    main()

