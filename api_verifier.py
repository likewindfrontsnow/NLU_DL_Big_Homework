# check_api.py
import requests
import sys
try:
    from config import ERNIE_CONFIG, ERNIE_API_KEY
except ImportError:
    print("❌ 错误：无法找到 'config.py' 文件或文件内容不完整。")
    print("   请确保 'config.py' 存在，并且定义了 ERNIE_API_KEY 和 ERNIE_CONFIG。")
    sys.exit(1)
except ValueError as e:
    print(f"❌ 错误: {e}")
    print("   请检查你的 .env 文件是否已正确设置 ERNIE_API_KEY。")
    sys.exit(1)


def check_api_connectivity(config: dict) -> bool:
    """
    检测 ERNIE API 的连通性、密钥有效性和模型可用性。
    """
    print("--- 正在检测 API 连通性与密钥有效性 ---")
    
    # 1. 检查传入的配置是否有效
    api_key = config.get('api_key')
    base_url = config.get('base_url')
    model = config.get('model')

    if not api_key:
        print("❌ 配置错误：ERNIE_CONFIG 中 'api_key' 为空。")
        return False
    if not base_url:
        print("❌ 配置错误：ERNIE_CONFIG 中 'base_url' 为空。")
        return False
    if not model:
        print("❌ 配置错误：ERNIE_CONFIG 中 'model' 为空。")
        return False

    with open("source_transcript.txt",'r',encoding='utf-8') as f:
        text=f.read()
        

    test_prompt = text;
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}", 
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": test_prompt}]
        }
        
        api_url = base_url + "/chat/completions"
        
        print(f"正在尝试连接到: {api_url}")
        print(f"使用模型: {model}")

        # 3. 发送请求
        response = requests.post(api_url, json=payload, headers=headers, timeout=120)

        # 4. 分析响应
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and result.get('choices', [{}])[0].get('message', {}).get('content'):
                print("✅ API 密钥有效，连通性良好！")
                print(f"   模型 ({model}) 返回示例: \"{result['choices'][0]['message']['content'][:30]}...\"")
                return True
            # 处理百度千帆特有的业务错误码
            elif 'error_code' in result:
                print(f"❌ API 业务错误 (HTTP 200 但包含错误): {result.get('error_msg')} (Code: {result.get('error_code')})")
                if result.get('error_code') == 336100: # 示例：模型不支持
                     print("   提示：这个错误码可能意味着 'model' 名称不正确或无权访问。")
                return False
            else:
                print("⚠️ API 响应成功 (HTTP 200)，但返回内容结构异常。请检查模型名称或配置。")
                print(f"   原始响应: {str(result)[:150]}...")
                return False
        
        elif response.status_code == 401:
            print("❌ API 密钥无效 (HTTP 401 Unauthorized)。")
            print("   请检查 .env 文件中的 ERNIE_API_KEY 是否正确，或者 API Key 是否已过期。")
            return False
            
        elif response.status_code == 404:
            print(f"❌ API 地址错误 (HTTP 404 Not Found)。")
            print(f"   请检查 ERNIE_CONFIG 中的 'base_url' ({base_url}) 是否正确。")
            return False

        else:
            print(f"❌ API 调用失败，HTTP 状态码: {response.status_code}")
            try:
                print(f"   错误详情: {response.json()}") # 尝试解析 JSON 错误
            except requests.exceptions.JSONDecodeError:
                print(f"   错误详情: {response.text[:150]}...") # 否则显示原始文本
            return False

    except requests.exceptions.Timeout:
        print("❌ 请求超时。请检查网络连接或 API 地址是否正确，以及是否需要设置网络代理。")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ 连接错误。无法连接到 API 地址 ({api_url})。")
        print(f"   详细信息: {e}")
        return False
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")
        return False

if __name__ == "__main__":
    if check_api_connectivity(ERNIE_CONFIG):
        print("\nAPI 验证通过。")
    else:
        print("\nAPI 验证失败，请在继续前修正 API 配置。")