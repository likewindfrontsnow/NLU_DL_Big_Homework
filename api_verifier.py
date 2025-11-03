import requests
import time

# 保持原有的配置结构
ERNIE_CONFIG = {
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_key": "bce-v3/ALTAK-Etj0lvrkCl7MGkkOTCeWr/805ca0828876600fd02683d34e348b2ed59f7a74",
    "model": "ernie-3.5-8k" 
}

def check_api_connectivity(config: dict) -> bool:
    print("--- 正在检测 API 连通性与密钥有效性 ---")
    
    test_prompt = "Hello" # 使用一个简单的测试输入
    
    try:
        headers = {
            "Authorization": f"Bearer {config.get('api_key')}", 
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": config.get('model'),
            "messages": [{"role": "user", "content": test_prompt}]
        }
        
        api_url = config.get('base_url') + "/chat/completions"
        
        response = requests.post(api_url, json=payload, headers=headers, timeout=10)

        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and result['choices'][0]['message']['content']:
                print("✅ API 密钥有效，连通性良好！")
                print(f"   模型 ({config.get('model')}) 返回示例: \"{result['choices'][0]['message']['content'][:20]}...\"")
                return True
            else:
                print("⚠️ API 响应成功 (HTTP 200)，但返回内容结构异常。请检查模型名称或配置。")
                return False
        
        elif response.status_code == 401:
            print("❌ API 密钥无效 (HTTP 401 Unauthorized)。请检查 ERNIE_CONFIG 中的 'api_key' 是否正确。")
            return False

        else:
            print(f"❌ API 调用失败，HTTP 状态码: {response.status_code}")
            print(f"   错误详情: {response.text[:100]}...")
            return False

    except requests.exceptions.Timeout:
        print("❌ 请求超时。请检查网络连接或 API 地址是否正确。")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ 连接错误。无法连接到 API 地址。")
        return False
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")
        return False

if __name__ == "__main__":
    if check_api_connectivity(ERNIE_CONFIG):
        print("\nAPI 验证通过")
    else:
        print("\nAPI 验证失败，请在继续前修正 API 配置。")