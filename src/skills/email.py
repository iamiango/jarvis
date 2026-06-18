"""Email Skill - 邮件发送技能

使用 aiosmtplib 实现异步 SMTP 邮件发送。
支持 Gmail、163、QQ、126 等国内外邮箱。
"""
from typing import Any, Dict, List, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os

from .base import BaseSkill, SkillOutput


class EmailSkill(BaseSkill):
    """Email Skill - 发送邮件

    支持:
    - 纯文本邮件
    - HTML 邮件
    - 带附件的邮件
    - 国内邮箱 (163、QQ、126等) 和国外邮箱 (Gmail等)
    """

    name = "send_email"
    description = "发送邮件到指定地址，支持文本、HTML 格式和附件"

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        use_tls: Optional[bool] = None,
        use_ssl: Optional[bool] = None,
        default_to: Optional[str] = None,
    ):
        """初始化 Email Skill

        Args:
            smtp_host: SMTP 服务器地址（默认从环境变量读取）
            smtp_port: SMTP 端口（465=SSL, 587=TLS, 25=无加密）
            smtp_user: SMTP 用户名（默认从环境变量读取）
            smtp_password: SMTP 密码/授权码（默认从环境变量读取）
            use_tls: 是否使用 STARTTLS（587端口常用）
            use_ssl: 是否使用 SSL（465端口常用，国内邮箱推荐）
            default_to: 默认收件人邮箱（未指定收件人时使用）
        """
        super().__init__()
        self.smtp_host = smtp_host or os.getenv("SMTP_HOST", "smtp.163.com")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "465"))
        self.smtp_user = smtp_user or os.getenv("SMTP_USER", "")
        self.smtp_password = smtp_password or os.getenv("SMTP_PASSWORD", "")
        self.default_to = default_to or os.getenv("SMTP_DEFAULT_TO", "")

        # 自动根据端口判断加密方式
        if use_ssl is not None:
            self.use_ssl = use_ssl
        else:
            self.use_ssl = os.getenv("SMTP_USE_SSL", "").lower() == "true" or self.smtp_port == 465

        if use_tls is not None:
            self.use_tls = use_tls
        else:
            self.use_tls = os.getenv("SMTP_USE_TLS", "").lower() == "true" or self.smtp_port == 587
        self.use_tls = use_tls

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "收件人邮箱地址，多个地址用逗号分隔"
                },
                "subject": {
                    "type": "string",
                    "description": "邮件主题"
                },
                "body": {
                    "type": "string",
                    "description": "邮件正文内容"
                },
                "html": {
                    "type": "boolean",
                    "description": "是否为 HTML 格式（默认 False）"
                },
                "cc": {
                    "type": "string",
                    "description": "抄送地址，多个地址用逗号分隔（可选）"
                },
                "bcc": {
                    "type": "string",
                    "description": "密送地址，多个地址用逗号分隔（可选）"
                },
                "attachments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "附件文件路径列表（可选）"
                },
            },
            "required": ["subject", "body"]
        }

    async def execute(
        self,
        subject: str,
        body: str,
        to: Optional[str] = None,
        html: bool = False,
        cc: Optional[str] = None,
        bcc: Optional[str] = None,
        attachments: Optional[List[str]] = None,
        **kwargs
    ) -> SkillOutput:
        """发送邮件

        Args:
            subject: 邮件主题
            body: 邮件正文
            to: 收件人地址（可选，未指定时使用默认收件人）
            html: 是否为 HTML 格式
            cc: 抄送地址
            bcc: 密送地址
            attachments: 附件路径列表

        Returns:
            SkillOutput 包含发送结果
        """
        # 使用默认收件人（如果未指定）
        recipient = to or self.default_to
        if not recipient:
            return SkillOutput(
                success=False,
                result=None,
                error="未指定收件人，且未配置默认收件人 (SMTP_DEFAULT_TO)"
            )

        # 检查 SMTP 配置
        if not self.smtp_user or not self.smtp_password:
            return SkillOutput(
                success=False,
                result=None,
                error="SMTP 配置不完整，请设置 SMTP_USER 和 SMTP_PASSWORD 环境变量"
            )

        try:
            import aiosmtplib

            # 解析收件人
            to_addrs = [addr.strip() for addr in recipient.split(",")]
            all_recipients = to_addrs.copy()

            # 创建邮件
            if attachments:
                msg = MIMEMultipart()
                content_type = "html" if html else "plain"
                msg.attach(MIMEText(body, content_type, "utf-8"))

                # 添加附件
                for filepath in attachments:
                    if os.path.exists(filepath):
                        with open(filepath, "rb") as f:
                            part = MIMEBase("application", "octet-stream")
                            part.set_payload(f.read())
                            encoders.encode_base64(part)
                            filename = os.path.basename(filepath)
                            part.add_header(
                                "Content-Disposition",
                                f"attachment; filename={filename}"
                            )
                            msg.attach(part)
            else:
                content_type = "html" if html else "plain"
                msg = MIMEText(body, content_type, "utf-8")

            # 设置邮件头
            msg["Subject"] = subject
            msg["From"] = self.smtp_user
            msg["To"] = ", ".join(to_addrs)

            if cc:
                cc_addrs = [addr.strip() for addr in cc.split(",")]
                msg["Cc"] = ", ".join(cc_addrs)
                all_recipients.extend(cc_addrs)

            if bcc:
                bcc_addrs = [addr.strip() for addr in bcc.split(",")]
                all_recipients.extend(bcc_addrs)

            # 发送邮件 - 根据配置选择 SSL 或 TLS
            if self.use_ssl:
                # SSL 连接 (465端口，国内邮箱常用)
                await aiosmtplib.send(
                    msg,
                    hostname=self.smtp_host,
                    port=self.smtp_port,
                    username=self.smtp_user,
                    password=self.smtp_password,
                    use_tls=True,  # 直接使用 TLS/SSL
                )
            else:
                # STARTTLS 连接 (587端口，Gmail 常用)
                await aiosmtplib.send(
                    msg,
                    hostname=self.smtp_host,
                    port=self.smtp_port,
                    username=self.smtp_user,
                    password=self.smtp_password,
                    start_tls=self.use_tls,
                )

            return SkillOutput(
                success=True,
                result={
                    "message": f"邮件已成功发送到 {recipient}",
                    "recipients": all_recipients,
                    "subject": subject,
                }
            )

        except ImportError:
            return SkillOutput(
                success=False,
                result=None,
                error="缺少 aiosmtplib 依赖，请运行: pip install aiosmtplib"
            )
        except Exception as e:
            return SkillOutput(
                success=False,
                result=None,
                error=f"邮件发送失败: {str(e)}"
            )


class EmailTemplateSkill(BaseSkill):
    """Email Template Skill - 使用模板发送邮件"""

    name = "send_template_email"
    description = "使用预定义模板发送邮件"

    def __init__(self, email_skill: Optional[EmailSkill] = None):
        super().__init__()
        self._email_skill = email_skill or EmailSkill()
        self._templates: Dict[str, Dict[str, str]] = {
            "stock_report": {
                "subject": "股票分析报告 - {symbol}",
                "body": """
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
        .header {{ background-color: #4CAF50; color: white; padding: 10px; text-align: center; }}
        .content {{ padding: 20px; }}
        .footer {{ background-color: #f1f1f1; padding: 10px; text-align: center; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>股票分析报告</h1>
        <h3>{symbol} - {stock_name}</h3>
    </div>
    <div class="content">
        {report_content}
    </div>
    <div class="footer">
        <p>此报告由 Jarvis AI 自动生成，仅供参考，不构成投资建议。</p>
        <p>生成时间: {timestamp}</p>
    </div>
</body>
</html>
"""
            }
        }

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "收件人邮箱地址"
                },
                "template": {
                    "type": "string",
                    "enum": list(self._templates.keys()),
                    "description": "邮件模板名称"
                },
                "variables": {
                    "type": "object",
                    "description": "模板变量字典"
                },
            },
            "required": ["to", "template", "variables"]
        }

    async def execute(
        self,
        to: str,
        template: str,
        variables: Dict[str, Any],
        **kwargs
    ) -> SkillOutput:
        """使用模板发送邮件"""
        if template not in self._templates:
            return SkillOutput(
                success=False,
                result=None,
                error=f"未知模板: {template}"
            )

        tpl = self._templates[template]
        try:
            subject = tpl["subject"].format(**variables)
            body = tpl["body"].format(**variables)
        except KeyError as e:
            return SkillOutput(
                success=False,
                result=None,
                error=f"模板变量缺失: {e}"
            )

        return await self._email_skill.execute(
            to=to,
            subject=subject,
            body=body,
            html=True,
        )
