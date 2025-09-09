import os
import time
import urllib.request
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import ElementClickInterceptedException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# === 설정 ===
CHROMEDRIVER_PATH = r"C:\\Users\\baby3\\Downloads\\chromedriver-win64\\chromedriver-win64\\chromedriver.exe"
SAVE_ROOT = r"C:\Users\baby3\OneDrive\바탕 화면\0720_사진(추가)"

# === 이미지 크롤링 및 저장 ===
def download_daum_images(query, save_dir, max_count=1000):
    os.makedirs(save_dir, exist_ok=True)

    options = Options()
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--window-size=1920,1080')

    driver = webdriver.Chrome(service=Service(CHROMEDRIVER_PATH), options=options)
    url = f"https://search.daum.net/search?w=img&q={query}"
    driver.get(url)
    time.sleep(3)
    print(f"🔍 '{query}' 접속 완료")

    for _ in range(15):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.5)

    thumbs = driver.find_elements(By.CSS_SELECTOR, "div.image_main div.wrap_thumb a")
    print(f"🖼️ 썸네일 수: {len(thumbs)}")

    count = 0
    for idx, thumb_link in enumerate(thumbs):
        try:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", thumb_link)
            time.sleep(0.5)

            try:
                thumb_link.click()
            except ElementClickInterceptedException:
                driver.execute_script("arguments[0].click();", thumb_link)
                time.sleep(1.0)

            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.image_viewer div.item_img img"))
            )

            viewer_img = driver.find_element(By.CSS_SELECTOR, "div.image_viewer div.item_img img")
            src = viewer_img.get_attribute("src")

            if src and src.startswith("http"):
                filename = f"{query.replace(' ', '_')}_{idx:03}.jpg"
                img_path = os.path.join(save_dir, filename)
                urllib.request.urlretrieve(src, img_path)
                print(f"✅ 저장됨: {filename}")
                count += 1
                if count >= max_count:
                    break
        except Exception as e:
            print(f"❌ [{idx}] 오류 발생: {e}")
            continue

    driver.quit()
    print(f"📦 최종 저장: {count}장 → {save_dir}")

# === 실행 ===
if __name__ == "__main__":
    brands = ["서브웨이 플랫브레드"]
    for brand in brands:
        print(f"\n=== 🍔 {brand} 시작 ===")
        save_path = os.path.join(SAVE_ROOT, brand.replace(" ", "_"))
        download_daum_images(brand, save_path, max_count=1000)
