import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 모델과 데이터로더 import
from dataloader_brand import get_brand_dataloaders
from model_brand_resnet import get_resnet18_brand_model

# 하이퍼파라미터 설정
batch_size = 32
num_epochs = 15
learning_rate = 1e-4
save_path = 'best_brand_model.pth'

def save_plot_as_image(epoch, epochs_list, train_losses, train_accuracies, val_losses, val_accuracies):
    """각 에포크마다 그래프를 이미지로 저장"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss 그래프
    ax1.plot(epochs_list, train_losses, 'b-', label='Train Loss', linewidth=3, marker='o', markersize=6)
    ax1.plot(epochs_list, val_losses, 'r-', label='Val Loss', linewidth=3, marker='s', markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, num_epochs)
    
    # Accuracy 그래프
    ax2.plot(epochs_list, train_accuracies, 'b-', label='Train Accuracy', linewidth=3, marker='o', markersize=6)
    ax2.plot(epochs_list, val_accuracies, 'r-', label='Val Accuracy', linewidth=3, marker='s', markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(1, num_epochs)
    
    # 현재 최고 성능 표시
    if val_accuracies:
        best_val_acc = max(val_accuracies)
        best_epoch = val_accuracies.index(best_val_acc) + 1
        ax2.axhline(y=best_val_acc, color='orange', linestyle='--', alpha=0.7)
        ax2.text(num_epochs*0.7, best_val_acc+0.02, f'Best: {best_val_acc:.4f} (Epoch {best_epoch})', 
                fontsize=10, fontweight='bold', color='orange')
    
    plt.suptitle(f'Brand Classification Training Progress - Epoch {epoch}/{num_epochs}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 에포크별 이미지 저장
    plt.savefig(f'training_plots_brand/epoch_{epoch:02d}.png', dpi=300, bbox_inches='tight')
    plt.close()

def update_plots(ax1, ax2, epochs_list, train_losses, train_accuracies, val_losses, val_accuracies):
    """실시간으로 그래프 업데이트"""
    ax1.clear()
    ax2.clear()
    
    # Loss 그래프
    ax1.plot(epochs_list, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o')
    ax1.plot(epochs_list, val_losses, 'r-', label='Val Loss', linewidth=2, marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy 그래프
    ax2.plot(epochs_list, train_accuracies, 'b-', label='Train Acc', linewidth=2, marker='o')
    ax2.plot(epochs_list, val_accuracies, 'r-', label='Val Acc', linewidth=2, marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)

# 🛡️ 메인 가드 - Windows 멀티프로세싱 오류 방지
if __name__ == '__main__':
    # 저장 폴더 생성
    os.makedirs("training_plots_brand", exist_ok=True)
    
    # 장치 설정 (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 사용 장치: {device}")
    
    # 데이터 불러오기
    train_loader, val_loader, _ = get_brand_dataloaders(
        data_dir=r"C:\Users\baby3\OneDrive\바탕 화면\0716_ 모델링\project_model\0716_split_data_test100\multi_classification",  # ← 사용자에 맞게 경로 수정
        batch_size=batch_size,
        num_workers=2
    )
    
    # 모델 불러오기
    model = get_resnet18_brand_model(num_classes=9, pretrained=True).to(device)
    
    # 손실 함수 & 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 📊 학습 기록을 위한 리스트
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    epochs_list = []
    
    # 🎨 그래프 스타일 설정
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    # 🎨 실시간 그래프 설정
    plt.ion()  # Interactive mode on
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 학습 루프
    best_val_acc = 0.0
    print("\n🚀 브랜드 분류 모델 학습 시작! (실시간 그래프 + 이미지 저장)")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        print(f"\n[Epoch {epoch+1}/{num_epochs}]")
        
        # 학습 단계
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct / total
        avg_train_loss = train_loss / total
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        # 검증 단계
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_acc = correct / total
        avg_val_loss = val_loss / total
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # 📊 기록 저장
        epochs_list.append(epoch + 1)
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        
        # 🎨 실시간 그래프 업데이트
        update_plots(ax1, ax2, epochs_list, train_losses, train_accuracies, val_losses, val_accuracies)
        
        # 📸 에포크별 그래프 이미지 저장
        save_plot_as_image(epoch + 1, epochs_list, train_losses, train_accuracies, val_losses, val_accuracies)
        
        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved to: {save_path}")
    
    print(f"\n🎉 학습 완료! 최고 검증 정확도: {best_val_acc:.4f}")
    print("=" * 60)
    
    # 📊 최종 그래프 저장
    plt.ioff()  # Interactive mode off
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, train_losses, 'b-', label='Train Loss', linewidth=3, marker='o', markersize=8)
    plt.plot(epochs_list, val_losses, 'r-', label='Val Loss', linewidth=3, marker='s', markersize=8)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('Final Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, train_accuracies, 'b-', label='Train Accuracy', linewidth=3, marker='o', markersize=8)
    plt.plot(epochs_list, val_accuracies, 'r-', label='Val Accuracy', linewidth=3, marker='s', markersize=8)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Final Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    plt.suptitle(f'Brand Classification Training Results - Best Val Acc: {best_val_acc:.4f}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('final_training_results_brand.png', dpi=300, bbox_inches='tight')
    print("\n📊 최종 학습 그래프 저장: final_training_results_brand.png")
    plt.show()
    
    # 📈 학습 기록을 CSV로 저장
    history_df = pd.DataFrame({
        'epoch': epochs_list,
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies
    })
    history_df.to_csv('training_history_brand.csv', index=False)
    print("📋 학습 기록 저장: training_history_brand.csv")
    
    # 📊 브랜드별 성능 분석 (옵션)
    print("\n📁 생성된 파일들:")
    print("   🖼️ training_plots_brand/ - 에포크별 그래프 이미지들")
    print("   📊 final_training_results_brand.png - 최종 결과 그래프")
    print("   📋 training_history_brand.csv - 수치 데이터")
    print("   💾 best_brand_model.pth - 최고 성능 모델 가중치")