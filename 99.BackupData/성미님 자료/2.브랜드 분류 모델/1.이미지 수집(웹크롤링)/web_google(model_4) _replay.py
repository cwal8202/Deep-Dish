import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from PIL import Image
import io
import random
import re

# 드라이버 경로 설정
driver_path = "C:/Users/baby3/Downloads/chromedriver-win64/chromedriver-win64/chromedriver.exe"



    # "롯데리아 햄버거 사진": "food/hamburger/lotteria",
    # "롯데리아 버거 실물": "food/hamburger/lotteria",
    # "롯데리아 불고기버거": "food/hamburger/lotteria",
    
    # "버거킹 햄버거 사진": "food/hamburger/burgerking",
    # "버거킹 와퍼 실물": "food/hamburger/burgerking", 
    # "버거킹 치킨버거": "food/hamburger/burgerking",
    
    # "맥도날드 햄버거 사진": "food/hamburger/mcdonalds",
    # "맥도날드 빅맥 실물": "food/hamburger/mcdonalds",
    # "맥도날드 치킨버거": "food/hamburger/mcdonalds",
    
    # "쉑쉑버거 사진": "food/hamburger/shakeshack",
    # "쉑쉑버거 실물": "food/hamburger/shakeshack",
    
    # "서브웨이 샌드위치 사진": "food/sandwich/subway",
    # "서브웨이 실물": "food/sandwich/subway",
    # "서브웨이 내돈내산": "food/sandwich/subway",

    # "홍루이젠 샌드위치 사진": "food/sandwich/hongruijian", 
    # "홍루이젠 실물": "food/sandwich/hongruijian",
    # "홍루이젠 내돈내산": "food/sandwich/hongruijian",
    
    # "노티드 도넛 사진": "food/dessert/knotted",
    # "노티드 도넛 실물": "food/dessert/knotted",
    # "노티드 도넛 선물": "food/dessert/knotted",
    
    # "몽슈슈 롤케이크 사진": "food/dessert/monchouchou",
    # "몽슈슈 케이크 실물": "food/dessert/monchouchou", 
    # "몽슈슈 선물": "food/dessert/monchouchou", 
    
    # "투썸플레이스 케이크 사진": "food/dessert/twosome",
    # "투썸플레이스 디저트 실물": "food/dessert/twosome",
    # "투썸플레이스 조각 케이크": "food/dessert/twosome"
# 검색 키워드
search_list = {
    # "에그드랍 샌드위치 사진": "food/sandwich/eggdrop",
    # "에그드랍 실물": "food/sandwich/eggdrop",
    # "에그드랍 내돈내산": "food/sandwich/eggdrop",
    # "에그드랍": "food/sandwich/eggdrop",
    # "에그드랍 메뉴": "food/sandwich/eggdrop",
    # "에그드랍 추천": "food/sandwich/eggdrop",
    "벽 메뉴판" : "not_food/menu_list",
    "가게 내부" : "not_food/store_wall",
    "가게 외관" : "not_food/store_wall",
    "편의점 빵 포장지" : "not_food/bread_poket",
    "가게 홍보 사진" : "not_food/store",
    "쇼핑몰" : "not_food/shopping",
    "티셔츠" : "not_food/shopping",
    "애니메이션" : "not_food/shopping",
}

def setup_driver():
    """웹드라이버 설정"""
    service = Service(driver_path)
    options = webdriver.ChromeOptions()
    
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-logging')
    options.add_argument('--log-level=3')
    options.add_argument('--window-size=1920,1080')
    
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def url_encode(text):
    """간단한 URL 인코딩"""
    try:
        import urllib.parse
        return urllib.parse.quote(text.encode('utf-8'))
    except ImportError:
        import urllib
        return urllib.quote(text.encode('utf-8'))
    except:
        # fallback
        return text.replace(' ', '%20')

def get_image_urls(driver, max_images=500):
    """이미지 URL 수집 - 개선된 버전"""
    all_urls = set()
    last_count = 0
    stagnant_count = 0
    
    print(f"🔍 이미지 URL 수집 시작 (목표: {max_images}개)")
    
    for scroll in range(30):  # 30번 스크롤로 증가
        print(f"📜 스크롤 {scroll+1}/30")
        
        # 방법 1: 모든 img 태그 찾기
        try:
            img_elements = driver.find_elements(By.TAG_NAME, "img")
            print(f"    페이지에서 {len(img_elements)}개 img 태그 발견")
            
            for img in img_elements:
                try:
                    # 다양한 속성에서 URL 추출
                    for attr in ['src', 'data-src', 'data-original', 'data-lazy-src', 'data-iurl']:
                        url = img.get_attribute(attr)
                        if url and url.startswith(('http://', 'https://')):
                            # 기본 필터링
                            if any(skip in url.lower() for skip in ['icon', 'logo', 'button', 'avatar', 'sprite']):
                                continue
                            
                            # 이미지 확장자 확인 또는 이미지 관련 URL
                            if (any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']) or
                                'images' in url.lower() or 'img' in url.lower()):
                                all_urls.add(url)
                except:
                    continue
        except Exception as e:
            print(f"    img 태그 수집 오류: {e}")
        
        # 방법 2: 페이지 소스에서 더 많은 패턴으로 URL 추출
        try:
            page_source = driver.page_source
            
            # 다양한 패턴으로 이미지 URL 찾기
            patterns = [
                r'https://[^"\s]*\.(?:jpg|jpeg|png|webp|gif)[^"\s]*',
                r'"(https://[^"]*(?:images|img)[^"]*)"',
                r'"ou":"(https://[^"]*)"',
                r'"ru":"(https://[^"]*)"',
                r'data-src="(https://[^"]*)"',
                r'src="(https://[^"]*\.(?:jpg|jpeg|png|webp|gif)[^"]*)"'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, page_source, re.IGNORECASE)
                for match in matches:
                    url = match if isinstance(match, str) else match[0] if match else ""
                    if url and url.startswith('http') and not any(skip in url.lower() for skip in ['icon', 'logo', 'button']):
                        all_urls.add(url)
        except Exception as e:
            print(f"    페이지 소스 수집 오류: {e}")
        
        # 방법 3: JavaScript로 동적 이미지 수집
        try:
            js_script = """
            var urls = [];
            
            // 모든 img 요소 수집
            document.querySelectorAll('img').forEach(function(img) {
                [img.src, img.dataset.src, img.dataset.original].forEach(function(url) {
                    if (url && url.startsWith('http') && 
                        (url.includes('.jpg') || url.includes('.jpeg') || url.includes('.png') || 
                         url.includes('.webp') || url.includes('images'))) {
                        urls.push(url);
                    }
                });
            });
            
            // 스타일 속성에서 background-image URL 수집
            document.querySelectorAll('*').forEach(function(el) {
                var style = window.getComputedStyle(el);
                var bgImg = style.backgroundImage;
                if (bgImg && bgImg !== 'none') {
                    var match = bgImg.match(/url\\("?(https?:[^"\\)]+)"?\\)/);
                    if (match && match[1]) {
                        urls.push(match[1]);
                    }
                }
            });
            
            return urls;
            """
            js_urls = driver.execute_script(js_script)
            if js_urls:
                all_urls.update(js_urls)
                print(f"    JavaScript로 {len(js_urls)}개 추가 URL 발견")
        except Exception as e:
            print(f"    JavaScript 수집 오류: {e}")
        
        current_count = len(all_urls)
        print(f"    현재 수집: {current_count}개")
        
        # 진행 상황 체크
        if current_count == last_count:
            stagnant_count += 1
            if stagnant_count >= 3:  # 3번 연속 변화없으면 더 적극적으로 스크롤
                print("    더 많은 이미지 로딩을 위해 적극적 스크롤...")
                for i in range(3):
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(1)
                    driver.execute_script("window.scrollTo(0, 0);")
                    time.sleep(1)
                stagnant_count = 0
        else:
            stagnant_count = 0
            last_count = current_count
        
        if current_count >= max_images:
            print(f"    목표 달성! ({current_count}개)")
            break
        
        # 스크롤 다운
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        
        # 추가 대기 시간 (이미지 로딩)
        time.sleep(1)
        
        # 더보기 버튼 클릭 시도 (더 다양한 패턴)
        try:
            more_button_selectors = [
                "//input[@value='결과 더보기']",
                "//button[contains(text(), '더보기')]",
                "//button[contains(text(), 'Show more')]",
                "//div[contains(text(), '더보기')]",
                "//*[contains(@class, 'more') or contains(@class, 'load')]//button",
                "//*[contains(text(), 'More results')]"
            ]
            
            for selector in more_button_selectors:
                try:
                    buttons = driver.find_elements(By.XPATH, selector)
                    if buttons:
                        driver.execute_script("arguments[0].click();", buttons[0])
                        print("    🔄 더보기 버튼 클릭")
                        time.sleep(3)
                        break
                except:
                    continue
        except:
            pass
        
        # 랜덤 스크롤 (Google 감지 회피)
        if scroll % 5 == 0:
            import random
            for _ in range(random.randint(1, 3)):
                driver.execute_script(f"window.scrollBy(0, {random.randint(200, 800)});")
                time.sleep(0.5)
    
    print(f"✅ 총 {len(all_urls)}개 URL 수집 완료")
    return list(all_urls)[:max_images]

def download_image(url, save_path):
    """개별 이미지 다운로드"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Referer': 'https://www.google.com/',
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # 이미지 데이터 검증
        img_data = response.content
        if len(img_data) < 1024:  # 1KB 미만 제외
            return False
            
        img = Image.open(io.BytesIO(img_data))
        
        # 크기 검증
        if img.width < 100 or img.height < 100:
            return False
        
        # 비율 검증
        aspect_ratio = img.width / img.height
        if aspect_ratio > 5.0 or aspect_ratio < 0.2:
            return False
        
        # RGB 변환
        if img.mode != 'RGB':
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            else:
                img = img.convert('RGB')
        
        # 저장
        img.save(save_path, 'JPEG', quality=85)
        return True
        
    except Exception as e:
        return False

def download_images(image_urls, save_dir, keyword, max_images=200):
    """이미지 다운로드 (순차)"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 기존 이미지 수 확인
    existing_files = [f for f in os.listdir(save_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    existing_count = len(existing_files)
    
    downloaded_count = 0
    failed_count = 0
    
    print(f"💾 이미지 다운로드 시작 (목표: {max_images}장)")
    
    for i, url in enumerate(image_urls):
        if downloaded_count >= max_images:
            break
            
        keyword_short = keyword.replace(' ', '_').replace('/', '_')[:15]
        save_path = os.path.join(save_dir, f"{keyword_short}_{existing_count + i + 1:05d}.jpg")
        
        if download_image(url, save_path):
            downloaded_count += 1
            if downloaded_count % 10 == 0:
                success_rate = (downloaded_count / (downloaded_count + failed_count) * 100) if (downloaded_count + failed_count) > 0 else 0
                print(f"  ✅ {downloaded_count}/{max_images} 완료 (성공률: {success_rate:.1f}%)")
        else:
            failed_count += 1
    
    success_rate = (downloaded_count / (downloaded_count + failed_count) * 100) if (downloaded_count + failed_count) > 0 else 0
    print(f"🎯 다운로드 완료: {downloaded_count}장 (성공률: {success_rate:.1f}%)")
    return downloaded_count

def crawl_keyword(keyword, save_dir, max_images=200):
    """키워드별 크롤링"""
    driver = setup_driver()
    
    try:
        # Google 이미지 검색 (간단한 URL 인코딩)
        encoded_keyword = keyword.replace(' ', '+')
        search_url = f"https://www.google.com/search?q={encoded_keyword}&tbm=isch&hl=ko"
        print(f"🌐 검색: {keyword}")
        
        driver.get(search_url)
        time.sleep(5)
        
        # 이미지 URL 수집
        image_urls = get_image_urls(driver, max_images=max_images*2)
        
        if not image_urls:
            print("❌ 수집된 URL이 없습니다.")
            return 0
        
        # 이미지 다운로드
        downloaded = download_images(image_urls, save_dir, keyword, max_images)
        return downloaded
        
    except Exception as e:
        print(f"❌ 크롤링 오류 ({keyword}): {e}")
        return 0
    finally:
        try:
            driver.quit()
        except:
            pass
        time.sleep(5)

def main():
    """메인 함수"""
    
    # 목표 설정 (현실적으로)
    brand_targets = {
        "not_food/menu_list": 8000,
        "not_food/store_wall": 8000,
        "not_food/shopping": 8000,
        "not_food/store": 8000,
        "not_food/bread_poket": 8000,
        # "food/hamburger/lotteria": 2000,
        # "food/hamburger/burgerking": 2000,
        # "food/hamburger/mcdonalds": 2000,
        # "food/hamburger/shakeshack": 2000,
        
        # "food/sandwich/subway": 5000,
        # "food/sandwich/eggdrop": 5000,
    #     "food/sandwich/hongruijian": 5000,
        
    #     "food/dessert/knotted": 5000,
    #     "food/dessert/monchouchou": 5000,
    #     "food/dessert/twosome": 5000
    }
    
    # 현재 상태 확인
    current_counts = {}
    for folder_path in brand_targets.keys():
        if os.path.exists(folder_path):
            existing_count = len([f for f in os.listdir(folder_path) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            current_counts[folder_path] = existing_count
        else:
            current_counts[folder_path] = 0
    
    total_target = sum(brand_targets.values())
    total_current = sum(current_counts.values())
    
    print("🚀 간단한 브랜드별 음식 이미지 크롤링 시작!")
    print(f"현재: {total_current}장 / 목표: {total_target}장")
    
    # 크롤링 실행
    for keyword, folder_path in search_list.items():
        if current_counts[folder_path] >= brand_targets[folder_path]:
            print(f"⏩ {folder_path} 브랜드 목표 달성 (건너뛰기)")
            continue
        
        remaining = brand_targets[folder_path] - current_counts[folder_path]
        download_count = min(remaining, 3000)  # 키워드당 최대 1000장
        
        brand_name = folder_path.split('/')[-1]
        category = folder_path.split('/')[1]
        
        print(f"\n" + "="*50)
        print(f"📂 {category} > {brand_name}")
        print(f"🔍 키워드: '{keyword}'")
        print(f"📊 현재: {current_counts[folder_path]}/{brand_targets[folder_path]}장")
        
        downloaded = crawl_keyword(keyword, folder_path, download_count)
        current_counts[folder_path] += downloaded
        
        if current_counts[folder_path] >= brand_targets[folder_path]:
            print(f"🎯 {brand_name} 브랜드 목표 달성!")
        
        print("⏳ 다음 검색까지 대기...")
        time.sleep(10)
    
    # 최종 결과
    print("\n" + "="*50)
    print("🎉 크롤링 완료!")
    print("📊 최종 결과:")
    
    total_downloaded = sum(current_counts.values())
    for folder_path in brand_targets.keys():
        brand = folder_path.split('/')[-1]
        category = folder_path.split('/')[1]
        count = current_counts[folder_path]
        target = brand_targets[folder_path]
        percentage = (count / target * 100) if target > 0 else 0
        print(f"  📁 {category}/{brand}: {count}/{target}장 ({percentage:.1f}%)")
    
    print(f"\n🎯 총 수집: {total_downloaded}/{total_target}장 ({total_downloaded/total_target*100:.1f}%)")

if __name__ == "__main__":
    main()