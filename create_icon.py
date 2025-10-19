#!/usr/bin/env python3
"""
Скрипт для создания иконки программы Imgdiff
Создает простую иконку с буквой "I" на синем фоне
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_icon():
    """Создает иконку 256x256 пикселей"""
    # Создаем изображение 256x256 с синим фоном
    size = 256
    img = Image.new('RGBA', (size, size), (25, 118, 210, 255))  # Синий цвет
    
    # Создаем объект для рисования
    draw = ImageDraw.Draw(img)
    
    # Добавляем белый круг в центре
    circle_center = size // 2
    circle_radius = size // 3
    draw.ellipse([
        circle_center - circle_radius,
        circle_center - circle_radius,
        circle_center + circle_radius,
        circle_center + circle_radius
    ], fill=(255, 255, 255, 255))
    
    # Добавляем букву "I" в центре
    try:
        # Пытаемся использовать системный шрифт
        font = ImageFont.truetype("arial.ttf", size // 4)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", size // 4)
        except:
            # Fallback на стандартный шрифт
            font = ImageFont.load_default()
    
    # Рисуем букву "I"
    text = "I"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = circle_center - text_width // 2
    y = circle_center - text_height // 2
    
    draw.text((x, y), text, fill=(25, 118, 210, 255), font=font)
    
    # Сохраняем как PNG
    img.save("imgdiff_icon.png")
    print("✅ Создана иконка: imgdiff_icon.png")
    
    # Конвертируем в ICO (если возможно)
    try:
        # Создаем несколько размеров для ICO
        sizes = [16, 32, 48, 64, 128, 256]
        icons = []
        
        for s in sizes:
            resized = img.resize((s, s), Image.Resampling.LANCZOS)
            icons.append(resized)
        
        # Сохраняем как ICO
        icons[0].save("imgdiff_icon.ico", format='ICO', sizes=[(s, s) for s in sizes])
        print("✅ Создана иконка: imgdiff_icon.ico")
        
    except Exception as e:
        print(f"⚠️ Не удалось создать ICO: {e}")
        print("📝 PNG иконка создана успешно")

if __name__ == "__main__":
    create_icon()
