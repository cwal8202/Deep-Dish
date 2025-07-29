from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import ElementClickInterceptedException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os, time, urllib.request

CHROMEDRIVER_PATH = r"C:\\Users\\baby3\\Downloads\\chromedriver-win64\\chromedriver-win64\\chromedriver.exe"

def download_daum_images(query, save_dir, max_count=50):
    os.makedirs(save_dir, exist_ok=True)

    options = Options()
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--window-size=1920,1080')

    try:
        driver = webdriver.Chrome(service=Service(CHROMEDRIVER_PATH), options=options)
        url = f"https://search.daum.net/search?w=img&q={query}"
        driver.get(url)
        time.sleep(3)

        print("✅ 페이지 접속 완료")

        # 스크롤 내려서 이미지 더 로딩
        for _ in range(15):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5)

        thumbs = driver.find_elements(By.CSS_SELECTOR, "div.image_main div.wrap_thumb a")
        print(f"🔎 썸네일 수: {len(thumbs)}")

        count = 0
        for idx, thumb_link in enumerate(thumbs):
            try:
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", thumb_link)
                time.sleep(0.5)

                try:
                    thumb_link.click()
                except ElementClickInterceptedException:
                    print(f"⚠️ [{idx}] 클릭 가로막힘 - 재시도 중...")
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
                    print(f"✅ 저장됨: {img_path}")
                    count += 1
                    if count >= max_count:
                        break
            except Exception as e:
                print(f"❌ [{idx}] 실패: {e}")
                continue

    except Exception as e:
        print(f"❌ 전체 실패: {e}")
    finally:
        try:
            driver.quit()
        except:
            pass
# 실행 부분
if __name__ == "__main__":
    brands = ['서브웨이 샌드위치', '에그드랍', '홍루이젠', '노티드 도넛', '몽슈슈 도지마롤', '투썸플레이스 조각케이크']
    
    for brand in brands:
        print(f"\n=== {brand} 시작 ===")
        download_daum_images(brand, f"./{brand}", max_count=2000)

