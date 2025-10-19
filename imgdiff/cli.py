"""
CLI интерфейс для imgdiff
"""
import sys
import argparse
from pathlib import Path
from typing import Optional

try:
    import typer
    from rich.console import Console
    from rich.progress import track
    TYPER_AVAILABLE = True
except ImportError:
    TYPER_AVAILABLE = False

import cv2
import numpy as np

from .core.diff import diff_mask_fast, coarse_to_fine
from .core.overlay import draw_diff_overlay, create_heatmap, draw_contours_on_image
from .core.io import safe_imread, safe_imwrite
from .core.morph import filter_small_components, dilate_mask


if TYPER_AVAILABLE:
    app = typer.Typer(help="Imgdiff - Быстрое сравнение изображений")
    console = Console()


def compare_images_core(
    img_a: np.ndarray,
    img_b: np.ndarray,
    fuzz: int = 10,
    use_lab: bool = True,
    use_coarse: bool = True,
    min_area: int = 50,
    thickness: int = 2,
    color: tuple = (0, 0, 255)
):
    """
    Ядро сравнения изображений (используется из CLI и GUI).
    """
    if use_coarse:
        # Многомасштабный подход
        boxes = coarse_to_fine(img_a, img_b, fuzz=fuzz, use_lab=use_lab, min_area=min_area)
        
        # Создаём полную маску из боксов
        mask = np.zeros(img_a.shape[:2], np.uint8)
        for x, y, w, h in boxes:
            roi_a = img_a[y:y+h, x:x+w]
            roi_b = img_b[y:y+h, x:x+w]
            roi_mask = diff_mask_fast(roi_a, roi_b, fuzz=fuzz, use_lab=use_lab)
            mask[y:y+h, x:x+w] = cv2.bitwise_or(mask[y:y+h, x:x+w], roi_mask)
    else:
        # Прямое сравнение
        mask = diff_mask_fast(img_a, img_b, fuzz=fuzz, use_lab=use_lab)
    
    # Фильтрация и расширение
    mask = filter_small_components(mask, min_area=min_area)
    mask = dilate_mask(mask, thickness=thickness)
    
    return mask


if TYPER_AVAILABLE:
    @app.command()
    def compare(
        image_a: Path = typer.Argument(..., help="Путь к первому изображению"),
        image_b: Path = typer.Argument(..., help="Путь ко второму изображению"),
        output: Path = typer.Option("diff.png", "--output", "-o", help="Путь для сохранения результата"),
        fuzz: int = typer.Option(10, "--fuzz", "-f", help="Порог различия (5-12 для Lab, 10-30 для RGB)"),
        use_lab: bool = typer.Option(True, "--lab/--rgb", help="Использовать Lab пространство"),
        use_coarse: bool = typer.Option(True, "--coarse/--direct", help="Многомасштабный подход"),
        min_area: int = typer.Option(50, "--min-area", "-m", help="Минимальная площадь региона"),
        thickness: int = typer.Option(2, "--thickness", "-t", help="Толщина линий выделения"),
        mode: str = typer.Option("contours", "--mode", help="Режим вывода: contours, overlay, heatmap"),
        color_r: int = typer.Option(0, "--color-r", help="Красный компонент цвета (0-255)"),
        color_g: int = typer.Option(0, "--color-g", help="Зелёный компонент цвета (0-255)"),
        color_b: int = typer.Option(255, "--color-b", help="Синий компонент цвета (0-255)"),
    ):
        """
        Сравнивает два изображения и сохраняет результат.
        """
        # Проверка путей
        if not image_a.exists():
            console.print(f"[red]Ошибка: файл {image_a} не найден[/red]")
            raise typer.Exit(1)
        
        if not image_b.exists():
            console.print(f"[red]Ошибка: файл {image_b} не найден[/red]")
            raise typer.Exit(1)
        
        # Загрузка
        console.print(f"Загрузка изображений...")
        img_a = safe_imread(str(image_a))
        img_b = safe_imread(str(image_b))
        
        if img_a is None or img_b is None:
            console.print("[red]Ошибка: не удалось загрузить изображения[/red]")
            raise typer.Exit(1)
        
        # Проверка размеров
        if img_a.shape != img_b.shape:
            console.print(f"[yellow]Предупреждение: размеры не совпадают ({img_a.shape} vs {img_b.shape})[/yellow]")
            console.print("Изменение размера изображения B...")
            img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
        
        # Сравнение
        console.print("Сравнение...")
        color_bgr = (color_b, color_g, color_r)  # RGB -> BGR
        mask = compare_images_core(img_a, img_b, fuzz, use_lab, use_coarse, min_area, thickness, color_bgr)
        
        # Создание результата
        if mode == "contours":
            result = draw_contours_on_image(img_b, mask, color_bgr, thickness)
        elif mode == "overlay":
            result = draw_diff_overlay(img_b, mask, color_bgr, alpha=0.6)
            # Конвертируем RGBA -> BGR для сохранения
            result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGR)
        elif mode == "heatmap":
            result = create_heatmap(img_a, img_b, use_lab=use_lab)
        else:
            console.print(f"[red]Неизвестный режим: {mode}[/red]")
            raise typer.Exit(1)
        
        # Сохранение
        console.print(f"Сохранение в {output}...")
        if safe_imwrite(str(output), result):
            # Подсчёт статистики
            diff_pixels = cv2.countNonZero(mask)
            total_pixels = mask.size
            diff_percent = (diff_pixels / total_pixels) * 100
            
            console.print(f"[green]Готово![/green]")
            console.print(f"Различающихся пикселей: {diff_pixels:,} ({diff_percent:.2f}%)")
        else:
            console.print(f"[red]Ошибка при сохранении результата[/red]")
            raise typer.Exit(1)


    @app.command()
    def batch(
        dir_a: Path = typer.Argument(..., help="Директория с первыми изображениями"),
        dir_b: Path = typer.Argument(..., help="Директория со вторыми изображениями"),
        output_dir: Path = typer.Argument(..., help="Директория для результатов"),
        fuzz: int = typer.Option(10, "--fuzz", "-f", help="Порог различия"),
        use_lab: bool = typer.Option(True, "--lab/--rgb", help="Использовать Lab пространство"),
        pattern: str = typer.Option("*.png", "--pattern", "-p", help="Паттерн файлов"),
    ):
        """
        Пакетное сравнение изображений из двух директорий.
        """
        if not dir_a.is_dir() or not dir_b.is_dir():
            console.print("[red]Ошибка: один из путей не является директорией[/red]")
            raise typer.Exit(1)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Получаем списки файлов
        files_a = sorted(dir_a.glob(pattern))
        files_b_dict = {f.name: f for f in dir_b.glob(pattern)}
        
        console.print(f"Найдено {len(files_a)} файлов в директории A")
        
        processed = 0
        errors = 0
        
        for file_a in track(files_a, description="Обработка..."):
            if file_a.name not in files_b_dict:
                console.print(f"[yellow]Пропуск {file_a.name}: нет пары в директории B[/yellow]")
                continue
            
            file_b = files_b_dict[file_a.name]
            output_file = output_dir / file_a.name
            
            try:
                img_a = safe_imread(str(file_a))
                img_b = safe_imread(str(file_b))
                
                if img_a is None or img_b is None:
                    raise Exception("Ошибка загрузки")
                
                if img_a.shape != img_b.shape:
                    img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
                
                mask = compare_images_core(img_a, img_b, fuzz=fuzz, use_lab=use_lab)
                result = draw_contours_on_image(img_b, mask, (0, 0, 255), 2)
                
                safe_imwrite(str(output_file), result)
                processed += 1
                
            except Exception as e:
                console.print(f"[red]Ошибка при обработке {file_a.name}: {e}[/red]")
                errors += 1
        
        console.print(f"\n[green]Обработано: {processed}, ошибок: {errors}[/green]")


def main():
    """Точка входа CLI"""
    if TYPER_AVAILABLE:
        app()
    else:
        # Fallback на простой argparse
        parser = argparse.ArgumentParser(description="Imgdiff - сравнение изображений")
        parser.add_argument("image_a", help="Первое изображение")
        parser.add_argument("image_b", help="Второе изображение")
        parser.add_argument("-o", "--output", default="diff.png", help="Выходной файл")
        parser.add_argument("-f", "--fuzz", type=int, default=10, help="Порог различия")
        
        args = parser.parse_args()
        
        img_a = safe_imread(args.image_a)
        img_b = safe_imread(args.image_b)
        
        if img_a is None or img_b is None:
            print("Ошибка загрузки изображений")
            sys.exit(1)
        
        if img_a.shape != img_b.shape:
            img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
        
        mask = compare_images_core(img_a, img_b, fuzz=args.fuzz)
        result = draw_contours_on_image(img_b, mask, (0, 0, 255), 2)
        
        safe_imwrite(args.output, result)
        print(f"Результат сохранён в {args.output}")


if __name__ == "__main__":
    main()


