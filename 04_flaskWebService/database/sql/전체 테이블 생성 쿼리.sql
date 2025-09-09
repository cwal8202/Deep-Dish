use deep_dish;

-- 업로드된 이미지
CREATE TABLE images (
    image_id BIGINT PRIMARY KEY AUTO_INCREMENT, -- 이미지 고유 ID
    image_url VARCHAR(1024) NOT NULL,          -- 이미지 저장 경로 또는 URL
    image_source VARCHAR(100),                       -- 이미지 출처 (예: 'instagram', 'user_upload')
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,        -- 수집(등록) 날짜. 추세 분석의 핵심
    INDEX idx_created_at (created_at)      -- 날짜 검색 속도 향상을 위한 인덱스
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 음식
CREATE TABLE menus (
menu_id INT PRIMARY KEY AUTO_INCREMENT,
menu_name VARCHAR(100) NOT NULL UNIQUE,
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
INDEX idx_menu_name (menu_name)
);

-- 재료
CREATE TABLE ingredients (
    ingredient_id INT PRIMARY KEY AUTO_INCREMENT,
    ingredient_name VARCHAR(100) NOT NULL UNIQUE,
    ingredient_category ENUM('main', 'seasoning') NOT NULL default 'main',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 메뉴별 식재료
CREATE TABLE menu_ingredients (
    menu_ingredient_id INT PRIMARY KEY AUTO_INCREMENT,
    menu_id INT NOT NULL,
    ingredient_id INT NOT NULL,
    amount FLOAT,                      -- 양 (숫자)
    unit VARCHAR(50),                  -- 단위 (문자열, 예: 'g', '개', '큰술')
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (menu_id) REFERENCES menus(menu_id) ON DELETE CASCADE,
    FOREIGN KEY (ingredient_id) REFERENCES ingredients(ingredient_id),
    UNIQUE KEY unique_menu_ingredient (menu_id, ingredient_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 메뉴별 레시피
CREATE TABLE menu_recipes (
recipe_id INT PRIMARY KEY AUTO_INCREMENT,
menu_id INT NOT NULL UNIQUE,
cooking_steps TEXT,
cooking_info TEXT,
flavor_characteristics TEXT,
nutrition_info TEXT,
serving_tips TEXT,
cooking_time_minutes INT,
difficulty_level ENUM('easy', 'medium', 'hard') DEFAULT 'medium',
servings INT DEFAULT 2,
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
FOREIGN KEY (menu_id) REFERENCES menus(menu_id) ON DELETE CASCADE
);

-- 브랜드
CREATE TABLE brands (
    brand_id INT PRIMARY KEY AUTO_INCREMENT,
    brand_name VARCHAR(255) NOT NULL UNIQUE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 분석 결과
CREATE TABLE analysis_results (
    result_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    image_id BIGINT NOT NULL,
    result_type ENUM('FOOD', 'BRAND') NOT NULL,
    detected_id INT NOT NULL,
    confidence_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 브랜드 카테고리
-- CREATE TABLE brand_categories (
--     brand_id INT NOT NULL,
--     middle_id INT NOT NULL,
--     PRIMARY KEY (brand_id, middle_id), -- 두 ID의 조합이 고유하도록 설정
--     FOREIGN KEY (brand_id) REFERENCES brands(brand_id) ON DELETE CASCADE,
--     FOREIGN KEY (middle_id) REFERENCES categories_middle(middle_id) ON DELETE CASCADE
-- ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 음식 분류 대, 중, 소, 메뉴
-- CREATE TABLE categories_major (
-- major_id INT PRIMARY KEY AUTO_INCREMENT,
-- major_name VARCHAR(50) NOT NULL UNIQUE,
-- created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
-- updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
-- );

-- CREATE TABLE categories_middle (
-- middle_id INT PRIMARY KEY AUTO_INCREMENT,
-- major_id INT NOT NULL,
-- middle_name VARCHAR(100) NOT NULL,
-- created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
-- updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
-- FOREIGN KEY (major_id) REFERENCES categories_major(major_id),
-- UNIQUE KEY unique_middle_per_major (major_id, middle_name) -- 대분류 + 중분류의 조합이 유니크함.
-- );

-- CREATE TABLE categories_minor (
-- minor_id INT PRIMARY KEY AUTO_INCREMENT,
-- middle_id INT NOT NULL,
-- minor_name VARCHAR(100) NOT NULL,
-- created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
-- updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
-- FOREIGN KEY (middle_id) REFERENCES categories_middle(middle_id),
-- UNIQUE KEY unique_minor_per_middle (middle_id, minor_name)
-- );

-- 영양정보
-- CREATE TABLE nutrition_facts (
--     nutrition_id INT PRIMARY KEY AUTO_INCREMENT,
--     menu_id INT NOT NULL UNIQUE,      -- menus 테이블과 1:1 관계
--     -- 이미지에 나온 주요 영양 성분들
--     calories_kcal INT,                -- 칼로리 (kcal)
--     protein_g FLOAT,                  -- 단백질 (g)
--     carbs_g FLOAT,                    -- 탄수화물 (g)
--     fat_g FLOAT,                      -- 지방 (g)
--     sugars_g FLOAT,                   -- 당분 (g)
--     sodium_mg FLOAT,                  -- 나트륨 (mg)
--     cholesterol_mg FLOAT,             -- 콜레스테롤 (mg)
--     dietary_fiber_g FLOAT,            -- 식이섬유 (g)
--     calcium_mg FLOAT,                 -- 칼슘 (mg)
--     iron_mg FLOAT,                    -- 철분 (mg)
--     zinc_mg FLOAT,                    -- 아연 (mg)
--     vitamin_a_mcg FLOAT,              -- 비타민 A (µg)
--     vitamin_c_mg FLOAT,               -- 비타민 C (mg)
--     caffeine_mg FLOAT,                -- 카페인 (mg)
--     -- 기타 특수 성분들
--     collagen_mg FLOAT,                -- 콜라겐 (mg)
--     taurine_mg FLOAT,                 -- 타우린 (mg)
--     omega3_mg FLOAT,                  -- 오메가3 (mg)
--     capsaicin_mg FLOAT,               -- 캡사이신 (mg)

--     FOREIGN KEY (menu_id) REFERENCES menus(menu_id) ON DELETE CASCADE
-- ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


