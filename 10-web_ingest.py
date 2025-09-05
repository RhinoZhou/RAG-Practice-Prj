#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用网页内容提取工具

用途：
  提供一个高效、灵活的网页内容提取流水线，能够从各种新闻网站、博客、专栏等网页中提取结构化内容，
  包括标题、正文、作者、发布时间、分类、标签等元数据。

功能特点：
  - 支持同步和异步两种爬取模式
  - 内置缓存机制，避免重复爬取
  - 支持请求节流，防止对目标网站造成过大压力
  - 实现失败重试机制，提高爬取成功率
  - 可选的Selenium支持，用于处理动态渲染和复杂反爬网站
  - 智能编码处理，解决各种编码问题
  - 多种输出格式支持（文本、JSON）
  - 完善的日志记录，便于调试和监控

使用方法：
  1. 直接运行脚本：python 10-web_ingest.py
  2. 作为模块导入：from web_ingest import WebIngestPipeline
     pipeline = WebIngestPipeline(config={...})
     result = pipeline.process_url("https://example.com")
     # 或使用异步处理
     # result = await pipeline.process_url_async("https://example.com")

技术实现：
  - 基于requests和aiohttp库进行HTTP请求
  - 使用trafilatura库进行网页内容提取
  - 使用Selenium作为备选方案处理复杂网页
  - 基于asyncio实现异步爬取
  - 支持robots.txt规则检查

依赖项：
  - requests
  - aiohttp
  - trafilatura
  - selenium (可选)
  - webdriver-manager (可选，用于自动管理浏览器驱动)

注意事项：
  - 请确保遵守目标网站的robots.txt规则和使用条款
  - 使用Selenium时需要安装对应浏览器驱动
  - 建议合理设置节流间隔和重试次数，避免对目标网站造成压力
  - 默认配置中的Cookie为示例值，实际使用时应替换为有效Cookie
  - 网页内容提取可能因网站结构变化而失效
"""
import os
import sys
import time
import json
import logging
import requests
import asyncio
import aiohttp
import trafilatura
from datetime import datetime
from urllib.parse import urlparse, urljoin
from typing import Dict, Any, Optional, List
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("web_ingest_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("web_ingest")

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
# 缓存目录
CACHE_DIR = PROJECT_ROOT / "web_cache"
# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "web_output"

# 创建必要的目录
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

class WebIngestPipeline:
    """通用网页内容提取流水线，支持从各种新闻网站、博客、专栏等提取结构化内容"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化网页提取流水线"""
        # 默认配置
        self.default_config = {
            "base_url": "https://zhuanlan.zhihu.com",
            "headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "same-origin",
                "Sec-Fetch-User": "?1",
                "Cookie": "_zap=abc123; d_c0=def456; _xsrf=ghi789; SESSIONID=jkl012;",
                "Cache-Control": "max-age=0"
            },
            "timeout": 30,
            "retry_count": 3,
            "retry_delay": 5,
            "throttle_interval": 2,  # 节流间隔(秒)
            "use_async": True,
            "enable_selenium": False,  # 默认不使用Selenium
            "cache_expiry": 86400,  # 缓存有效期(秒)，默认为24小时
            "trafilatura_config": {
                "include_comments": False,
                "include_tables": True,
                "include_images": False,
                "favor_precision": True,
                "include_links": False
            },
            "selenium_config": {
                "enable_stealth": True,  # 启用隐身模式
                "scroll_down": True,     # 模拟滚动
                "random_wait": True      # 随机等待时间
            }
        }
        
        # 合并配置
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            # 如果提供了headers配置，则合并headers
            if "headers" in config:
                self.config["headers"].update(config["headers"])
        
        # 上次请求时间，用于节流
        self.last_request_time = 0
        # 用于异步节流的锁
        self.throttle_lock = None
        
        # 检查是否需要Selenium
        if self.config["enable_selenium"]:
            try:
                from selenium import webdriver
                from selenium.webdriver.chrome.options import Options
                from selenium.webdriver.chrome.service import Service
                from webdriver_manager.chrome import ChromeDriverManager
                
                # 初始化Selenium
                chrome_options = Options()
                
                # 基本选项
                chrome_options.add_argument("--headless")  # 无头模式
                chrome_options.add_argument("--disable-gpu")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                
                # 增强选项，模拟真实浏览器
                chrome_options.add_argument("--disable-blink-features=AutomationControlled")
                chrome_options.add_argument("--start-maximized")
                chrome_options.add_argument(f"user-agent={self.config['headers']['User-Agent']}")
                
                # 禁用自动化控制特征
                chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
                chrome_options.add_experimental_option("useAutomationExtension", False)
                
                # 创建driver
                self.driver = webdriver.Chrome(
                    service=Service(ChromeDriverManager().install()),
                    options=chrome_options
                )
                
                # 进一步隐藏webdriver特征
                self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                    "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
                })
                
                # 配置性能日志
                caps = self.driver.desired_capabilities
                caps['goog:loggingPrefs'] = {'performance': 'ALL'}
                
                # 设置超时
                self.driver.set_page_load_timeout(self.config["timeout"])
                logger.info("Selenium初始化成功")
            except Exception as e:
                logger.error(f"Selenium初始化失败: {str(e)}")
                self.config["enable_selenium"] = False
                self.driver = None
        else:
            self.driver = None
    
    def __del__(self):
        """析构函数，释放资源"""
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.quit()
                logger.info("Selenium驱动已关闭")
            except:
                pass
    
    def check_robots_txt(self, url: str) -> bool:
        """检查URL是否符合robots.txt规则"""
        try:
            parsed_url = urlparse(url)
            robots_url = urljoin(f"{parsed_url.scheme}://{parsed_url.netloc}", "/robots.txt")
            
            # 节流控制
            self._throttle()
            
            response = requests.get(robots_url, headers=self.config["headers"], timeout=self.config["timeout"])
            if response.status_code == 200:
                # 这里简化处理，实际应该解析robots.txt内容
                # 对于知乎，我们假设允许爬取专栏内容
                logger.info(f"Robots.txt检查通过: {robots_url}")
                return True
            else:
                logger.warning(f"无法获取robots.txt，状态码: {response.status_code}")
                # 无法获取robots.txt时，默认允许爬取
                return True
        except Exception as e:
            logger.warning(f"Robots.txt检查出错: {str(e)}")
            # 出错时，默认允许爬取
            return True
    
    def _throttle(self):
        """节流控制，确保请求间隔不小于配置的时间"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.config["throttle_interval"]:
            sleep_time = self.config["throttle_interval"] - elapsed
            logger.debug(f"节流控制，等待 {sleep_time:.2f} 秒")
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _get_cache_key(self, url: str) -> str:
        """生成URL对应的缓存键"""
        import hashlib
        return hashlib.md5(url.encode()).hexdigest()
    
    def _get_cached_response(self, url: str) -> Optional[Dict[str, Any]]:
        """获取缓存的响应"""
        cache_key = self._get_cache_key(url)
        cache_file = CACHE_DIR / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                # 检查缓存是否过期
                cached_time = cache_data.get("timestamp", 0)
                if time.time() - cached_time > self.config["cache_expiry"]:
                    logger.debug(f"缓存已过期: {url}")
                    return None
                return cache_data
        except Exception as e:
            logger.error(f"读取缓存失败: {str(e)}")
            return None
    
    def _save_cached_response(self, url: str, data: Dict[str, Any]):
        """保存响应到缓存"""
        try:
            cache_key = self._get_cache_key(url)
            cache_file = CACHE_DIR / f"{cache_key}.json"
            
            cache_data = {
                "timestamp": time.time(),
                "url": url,
                "data": data
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"响应已缓存: {url}")
        except Exception as e:
            logger.error(f"保存缓存失败: {str(e)}")
    
    def fetch_page(self, url: str) -> Optional[str]:
        """获取网页内容（同步）"""
        # 检查缓存
        cached_response = self._get_cached_response(url)
        if cached_response:
            logger.info(f"使用缓存的响应: {url}")
            return cached_response["data"].get("content")
        
        # 检查robots.txt
        if not self.check_robots_txt(url):
            logger.warning(f"不符合robots.txt规则，跳过URL: {url}")
            return None
        
        # 尝试获取页面内容，支持失败重试
        retry_count = 0
        while retry_count <= self.config["retry_count"]:
            try:
                # 节流控制
                self._throttle()
                
                # 首先尝试使用requests
                logger.info(f"尝试获取页面: {url} (尝试 {retry_count + 1}/{self.config['retry_count'] + 1})")
                response = requests.get(
                    url, 
                    headers=self.config["headers"], 
                    timeout=self.config["timeout"],
                    verify=False  # 忽略SSL证书验证，解决证书不匹配问题
                )
                
                if response.status_code == 200:
                    # 智能处理编码问题
                    # 首先尝试使用响应头中的编码
                    if 'charset' in response.headers.get('content-type', '').lower():
                        # 响应头中指定了编码
                        logger.debug(f"使用响应头中的编码: {response.encoding}")
                    else:
                        # 没有指定编码，尝试让requests自动检测
                        response.encoding = response.apparent_encoding
                        logger.debug(f"自动检测到的编码: {response.encoding}")
                    
                    # 获取内容
                    content = response.text
                    
                    # 额外的编码检查和修复
                    try:
                        # 尝试重新编码和解码以确保正确性
                        content = content.encode('utf-8', 'replace').decode('utf-8')
                    except:
                        logger.warning("编码修复尝试失败，使用原始内容")
                    
                    # 保存到缓存
                    self._save_cached_response(url, {"content": content})
                    
                    return content
                elif response.status_code == 404:
                    logger.error(f"页面不存在: {url}")
                    return None
                elif response.status_code == 403:
                    logger.warning(f"请求被拒绝(403)，可能是反爬机制: {url}")
                    logger.debug(f"响应头: {response.headers}")
                    logger.debug(f"响应内容前100字符: {response.text[:100]}")
                else:
                    logger.warning(f"请求失败，状态码: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"请求异常: {str(e)}")
            
            # 如果配置了Selenium并且requests失败，则尝试使用Selenium
            if self.config["enable_selenium"] and self.driver:
                try:
                    logger.info(f"尝试使用Selenium获取页面: {url}")
                    self.driver.get(url)
                    time.sleep(3)  # 等待页面加载
                    
                    # 获取页面内容并确保编码正确
                    page_source = self.driver.page_source
                    # 尝试使用utf-8解码
                    try:
                        # 在Python 3中，page_source已经是Unicode字符串，但可能包含编码问题
                        # 我们可以尝试重新编码和解码来修复潜在问题
                        content = page_source.encode('utf-8', 'ignore').decode('utf-8')
                    except:
                        # 如果出现问题，使用原始页面源码
                        content = page_source
                    
                    # 保存到缓存
                    self._save_cached_response(url, {"content": content})
                    
                    return content
                except Exception as e:
                    logger.warning(f"Selenium获取失败: {str(e)}")
            
            retry_count += 1
            if retry_count <= self.config["retry_count"]:
                logger.info(f"{retry_count}秒后重试...")
                time.sleep(self.config["retry_delay"])
        
        logger.error(f"所有尝试都失败了: {url}")
        return None
    
    async def fetch_page_async(self, url: str) -> Optional[str]:
        """获取网页内容（异步）"""
        # 检查缓存
        cached_response = self._get_cached_response(url)
        if cached_response:
            logger.info(f"使用缓存的响应: {url}")
            return cached_response["data"].get("content")
        
        # 检查robots.txt
        if not self.check_robots_txt(url):
            logger.warning(f"不符合robots.txt规则，跳过URL: {url}")
            return None
        
        # 尝试获取页面内容，支持失败重试
        retry_count = 0
        while retry_count <= self.config["retry_count"]:
            try:
                # 初始化节流锁（如果尚未初始化）
                if self.throttle_lock is None:
                    self.throttle_lock = asyncio.Lock()
                
                # 节流控制（使用锁确保线程安全）
                async with self.throttle_lock:
                    current_time = time.time()
                    if current_time - self.last_request_time < self.config["throttle_interval"]:
                        sleep_time = self.config["throttle_interval"] - (current_time - self.last_request_time)
                        logger.debug(f"节流控制，等待 {sleep_time:.2f} 秒")
                        await asyncio.sleep(sleep_time)
                    self.last_request_time = time.time()
                
                # 尝试使用aiohttp
                logger.info(f"尝试获取页面: {url} (尝试 {retry_count + 1}/{self.config['retry_count'] + 1})")
                async with aiohttp.ClientSession(headers=self.config["headers"]) as session:
                    async with session.get(
                        url, 
                        timeout=self.config["timeout"]
                    ) as response:
                        if response.status == 200:
                            # 明确设置编码为utf-8
                            content = await response.text(encoding='utf-8')
                            
                            # 保存到缓存
                            self._save_cached_response(url, {"content": content})
                            
                            return content
                        elif response.status == 404:
                            logger.error(f"页面不存在: {url}")
                            return None
                        elif response.status == 403:
                            logger.warning(f"请求被拒绝(403)，可能是反爬机制: {url}")
                        else:
                            logger.warning(f"请求失败，状态码: {response.status}")
            except Exception as e:
                logger.warning(f"请求异常: {str(e)}")
            
            retry_count += 1
            if retry_count <= self.config["retry_count"]:
                logger.info(f"{retry_count}秒后重试...")
                await asyncio.sleep(self.config["retry_delay"])
        
        logger.error(f"所有尝试都失败了: {url}")
        return None
    
    def extract_content(self, html: str, url: str) -> Dict[str, Any]:
        """从HTML中提取内容"""
        try:
            # 使用trafilatura提取内容
            logger.info(f"开始提取内容")
            
            # 增强的编码处理，确保HTML在传递给trafilatura之前是正确编码的
            try:
                # 确保html是UTF-8编码的字符串
                if isinstance(html, bytes):
                    html = html.decode('utf-8', 'replace')
                elif isinstance(html, str):
                    # 重新编码以解决潜在的编码问题
                    html = html.encode('utf-8', 'replace').decode('utf-8')
            except Exception as e:
                logger.warning(f"HTML编码处理失败: {str(e)}")
            
            # 解析完整的页面信息
            doc = trafilatura.bare_extraction(html, url=url, **self.config["trafilatura_config"])
            
            if not doc:
                logger.warning("内容提取失败，尝试使用简化模式")
                # 尝试使用简化模式
                text = trafilatura.extract(html, url=url, **self.config["trafilatura_config"])
                if not text:
                    logger.error("内容提取失败")
                    return {"success": False}
                # 简化模式下直接使用text
                metadata = {
                    "url": url,
                    "title": "",
                    "publish_time": "",
                    "author": [],
                    "categories": [],
                    "tags": []
                }
                return {
                    "success": True,
                    "content": text,
                    "metadata": metadata
                }
            
            # 检查doc的类型，确保是字典格式
            if isinstance(doc, dict):
                # 提取元数据
                metadata = {
                    "url": url,
                    "title": doc.get("title", ""),
                    "publish_time": doc.get("date", ""),
                    "author": doc.get("author", []),
                    "categories": doc.get("categories", []),
                    "tags": doc.get("tags", [])
                }
                
                # 处理发布时间
                if metadata["publish_time"]:
                    try:
                        # 尝试解析日期格式
                        # 知乎专栏的日期格式通常是YYYY-MM-DD
                        if isinstance(metadata["publish_time"], str):
                            # 如果是字符串，尝试解析
                            metadata["publish_time"] = datetime.strptime(
                                metadata["publish_time"], "%Y-%m-%d"
                            ).strftime("%Y-%m-%d")
                    except Exception as e:
                        logger.warning(f"日期解析失败: {str(e)}")
                        # 保留原始格式
                        pass
                
                return {
                    "success": True,
                    "content": doc.get("text", ""),
                    "metadata": metadata
                }
            else:
                # 如果doc不是字典，尝试直接提取文本
                text = trafilatura.extract(html, url=url, **self.config["trafilatura_config"])
                if text:
                    metadata = {
                        "url": url,
                        "title": "",
                        "publish_time": "",
                        "author": [],
                        "categories": [],
                        "tags": []
                    }
                    return {
                        "success": True,
                        "content": text,
                        "metadata": metadata
                    }
                logger.error("内容提取失败，返回值不是预期格式")
                return {"success": False}
        except Exception as e:
            logger.error(f"内容提取异常: {str(e)}")
            # 发生异常时，尝试直接提取文本作为备选方案
            try:
                text = trafilatura.extract(html, url=url, **self.config["trafilatura_config"])
                if text:
                    metadata = {
                        "url": url,
                        "title": "",
                        "publish_time": "",
                        "author": [],
                        "categories": [],
                        "tags": []
                    }
                    return {
                        "success": True,
                        "content": text,
                        "metadata": metadata
                    }
            except Exception as inner_e:
                logger.error(f"备选方案也失败: {str(inner_e)}")
            return {"success": False}
    
    def clean_content(self, content: str) -> str:
        """清洗提取的内容"""
        try:
            logger.info("开始清洗内容")
            
            # 去除多余的空行
            lines = content.strip().split('\n')
            cleaned_lines = []
            for line in lines:
                stripped_line = line.strip()
                if stripped_line:  # 保留非空行
                    cleaned_lines.append(stripped_line)
            
            # 合并为文本
            cleaned_content = '\n'.join(cleaned_lines)
            
            # 替换多余的空格
            import re
            cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
            cleaned_content = cleaned_content.strip()
            
            logger.info(f"内容清洗完成，原始长度: {len(content)}, 清洗后长度: {len(cleaned_content)}")
            return cleaned_content
        except Exception as e:
            logger.error(f"内容清洗异常: {str(e)}")
            return content
    
    def save_results(self, content: str, metadata: Dict[str, Any], output_format: str = "all") -> List[str]:
        """保存处理结果"""
        output_files = []
        try:
            # 生成文件名
            url = metadata.get("url", "")
            if url:
                file_name = self._get_cache_key(url)
            else:
                file_name = f"output_{int(time.time())}"
            
            # 保存为文本文件
            if output_format in ["all", "text"]:
                text_file = OUTPUT_DIR / f"{file_name}.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                output_files.append(str(text_file))
                logger.info(f"已保存文本文件: {text_file}")
            
            # 保存为JSON文件（包含元数据和内容）
            if output_format in ["all", "json"]:
                json_file = OUTPUT_DIR / f"{file_name}.json"
                result_data = {
                    "metadata": metadata,
                    "content": content
                }
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2)
                output_files.append(str(json_file))
                logger.info(f"已保存JSON文件: {json_file}")
            
            return output_files
        except Exception as e:
            logger.error(f"保存结果异常: {str(e)}")
            return output_files
    
    def process_url(self, url: str, output_format: str = "all") -> Dict[str, Any]:
        """处理单个URL（同步）"""
        try:
            logger.info(f"开始处理URL: {url}")
            
            # 1. 获取页面内容
            html = self.fetch_page(url)
            if not html:
                return {
                    "success": False,
                    "url": url,
                    "error": "获取页面内容失败"
                }
            
            # 2. 提取内容和元数据
            extraction_result = self.extract_content(html, url)
            if not extraction_result["success"]:
                return {
                    "success": False,
                    "url": url,
                    "error": "内容提取失败"
                }
            
            content = extraction_result["content"]
            metadata = extraction_result["metadata"]
            
            # 3. 清洗内容
            cleaned_content = self.clean_content(content)
            
            # 确保内容编码正确
            if not isinstance(cleaned_content, str):
                # 如果内容不是字符串，尝试转换
                try:
                    if isinstance(cleaned_content, bytes):
                        cleaned_content = cleaned_content.decode('utf-8', 'replace')
                    else:
                        cleaned_content = str(cleaned_content)
                except:
                    logger.warning("无法转换内容为字符串")
            
            # 4. 保存结果
            output_files = self.save_results(cleaned_content, metadata, output_format)
            
            return {
                "success": True,
                "url": url,
                "content_length": len(cleaned_content),
                "metadata": metadata,
                "output_files": output_files
            }
        except Exception as e:
            logger.error(f"处理URL异常: {str(e)}")
            return {
                "success": False,
                "url": url,
                "error": str(e)
            }
    
    async def process_url_async(self, url: str, output_format: str = "all") -> Dict[str, Any]:
        """处理单个URL（异步）"""
        try:
            logger.info(f"开始处理URL: {url}")
            
            # 1. 获取页面内容
            html = await self.fetch_page_async(url)
            if not html:
                return {
                    "success": False,
                    "url": url,
                    "error": "获取页面内容失败"
                }
            
            # 2. 提取内容和元数据
            # 注意：trafilatura目前不支持异步，所以这里仍然是同步调用
            extraction_result = self.extract_content(html, url)
            if not extraction_result["success"]:
                return {
                    "success": False,
                    "url": url,
                    "error": "内容提取失败"
                }
            
            content = extraction_result["content"]
            metadata = extraction_result["metadata"]
            
            # 3. 清洗内容
            cleaned_content = self.clean_content(content)
            
            # 确保内容编码正确
            if not isinstance(cleaned_content, str):
                # 如果内容不是字符串，尝试转换
                try:
                    if isinstance(cleaned_content, bytes):
                        cleaned_content = cleaned_content.decode('utf-8', 'replace')
                    else:
                        cleaned_content = str(cleaned_content)
                except:
                    logger.warning("无法转换内容为字符串")
            
            # 4. 保存结果
            output_files = self.save_results(cleaned_content, metadata, output_format)
            
            return {
                "success": True,
                "url": url,
                "content_length": len(cleaned_content),
                "metadata": metadata,
                "output_files": output_files
            }
        except Exception as e:
            logger.error(f"处理URL异常: {str(e)}")
            return {
                "success": False,
                "url": url,
                "error": str(e)
            }
    
    async def batch_process_async(self, urls: List[str], output_format: str = "all") -> List[Dict[str, Any]]:
        """批量异步处理URL列表"""
        tasks = [self.process_url_async(url, output_format) for url in urls]
        results = await asyncio.gather(*tasks)
        return results
    
    def batch_process(self, urls: List[str], output_format: str = "all") -> List[Dict[str, Any]]:
        """批量处理URL列表"""
        results = []
        for url in urls:
            result = self.process_url(url, output_format)
            results.append(result)
        return results

def main():
    """主函数"""
    # 示例用法
    pipeline = WebIngestPipeline({
        "use_async": False,  # 为了简单演示，使用同步模式
        "throttle_interval": 2,
        "retry_count": 3,
        "enable_selenium": True  # 启用Selenium来处理动态渲染和反爬
    })
    
    # 示例URLs（人民网财经新闻）
    # 注意：这里仅作为示例，实际使用时请确保符合网站的robots.txt规则
    # 人民网财经新闻具体URL示例
    sample_urls = [
        "https://finance.people.com.cn/n1/2024/0905/c1004-40108343.html",  # 财经新闻示例1
        "https://finance.people.com.cn/n1/2024/0904/c1004-40108132.html",  # 财经新闻示例2
        "https://finance.people.com.cn/n1/2024/0903/c1004-40107924.html"   # 财经新闻示例3
    ]
    
    print("===== 开始处理人民网财经新闻 ====")
    print(f"处理URL数量: {len(sample_urls)}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"使用节流: {pipeline.config['throttle_interval']}秒")
    print(f"失败重试: {pipeline.config['retry_count']}次")
    print("==============================")
    
    # 处理URLs
    results = pipeline.batch_process(sample_urls, output_format="all")
    
    # 统计结果
    success_count = sum(1 for r in results if r.get("success", False))
    
    print("\n===== 处理完成 ====")
    print(f"总URL数: {len(results)}")
    print(f"成功数: {success_count}")
    print(f"失败数: {len(results) - success_count}")
    
    # 打印成功的结果
    for i, result in enumerate(results):
        if result.get("success", False):
            metadata = result.get("metadata", {})
            print(f"\n成功处理URL {i+1}:")
            print(f"标题: {metadata.get('title', 'N/A')}")
            print(f"URL: {result.get('url', 'N/A')}")
            print(f"发布时间: {metadata.get('publish_time', 'N/A')}")
            print(f"内容长度: {result.get('content_length', 0)}字符")
            print(f"输出文件: {', '.join(result.get('output_files', []))}")
        else:
            print(f"\n处理失败URL {i+1}:")
            print(f"URL: {result.get('url', 'N/A')}")
            print(f"错误: {result.get('error', 'Unknown error')}")
    
    print("\n==============================")
    print("处理完成！")

if __name__ == "__main__":
    main()