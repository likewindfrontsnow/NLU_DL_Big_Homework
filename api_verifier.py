# check_api.py
import requests
import sys

try:
    from config import LLM_CONFIG
except ImportError:
    print("❌ 错误：无法找到 'config.py' 文件。")
    print("   请确保 'config.py' 存在于同一目录下。")
    sys.exit(1)
except ValueError as e:
    print(f"❌ 配置加载错误: {e}")
    print("   请检查你的 .env 文件是否已正确设置 LLM_API_KEY, LLM_BASE_URL, 和 LLM_MODEL。")
    sys.exit(1)


def check_api_connectivity(config: dict) -> bool:
    """
    检测通用 LLM API 的连通性、密钥有效性和模型可用性。
    """
    print("--- 正在检测 API 连通性与密钥有效性 ---")
    
    # 1. 检查传入的通用配置
    api_key = config.get('api_key')
    base_url = config.get('base_url')
    model = config.get('model')
    provider_name = config.get('provider_name', 'LLM')

    if not api_key:
        print("❌ 配置错误：LLM_CONFIG 中 'api_key' 为空。")
        return False
    if not base_url:
        print("❌ 配置错误：LLM_CONFIG 中 'base_url' 为空。")
        return False
    if not model:
        print("❌ 配置错误：LLM_CONFIG 中 'model' 为空。")
        return False

    # 2. 准备测试请求
    #    我们使用一个简单的 "hello" 来测试连通性。
    #    这使得脚本更健壮，不依赖本地文件。
    #    如果你需要测试长文本，可以手动把长文本粘贴到这里。
    test_prompt = "hello"
    
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
        
        print(f"正在尝试连接到: {api_url} (服务商: {provider_name})")
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
            # 增加对 OpenAI 兼容错误的处理
            elif 'error' in result: 
                print(f"❌ API 业务错误: {result.get('error')}")
                return False
            elif 'error_code' in result:
                print(f"❌ API 业务错误: {result.get('error_msg')} (Code: {result.get('error_code')})")
                return False
            else:
                print("⚠️ API 响应成功 (HTTP 200)，但返回内容结构异常。")
                print(f"   原始响应: {str(result)[:150]}...")
                return False
        
        elif response.status_code == 401:
            print("❌ API 密钥无效 (HTTP 401 Unauthorized)。")
            print("   请检查 .env 文件中的 LLM_API_KEY 是否正确或已过期。")
            return False
            
        elif response.status_code == 404:
            print(f"❌ API 地址错误 (HTTP 404 Not Found)。")
            print(f"   请检查 .env 文件中的 LLM_BASE_URL ({base_url}) 是否正确。")
            return False

        else:
            print(f"❌ API 调用失败，HTTP 状态码: {response.status_code}")
            try:
                print(f"   错误详情: {response.json()}") 
            except requests.exceptions.JSONDecodeError:
                print(f"   错误详情: {response.text[:150]}...")
            return False

    except requests.exceptions.Timeout:
        print(f"❌ 请求超时 (超过 {120} 秒)。请检查网络连接或 API 地址是否正确。")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ 连接错误。无法连接到 API 地址 ({api_url})。")
        print(f"   详细信息: {e}")
        return False
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")
        return False

if __name__ == "__main__":
    if check_api_connectivity(LLM_CONFIG):
        print("\nAPI 验证通过。")
    else:
        print("\nAPI 验证失败，请在继续前修正 API 配置。")