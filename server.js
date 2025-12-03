const express = require('express');
const multer = require('multer');
const fs = require('fs').promises;
const path = require('path');
const cors = require('cors');

const app = express();
const port = 3000;

// 中间件
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// 确保目录存在
const quotationsDir = './assets/quotations';
const jsonFilePath = './assets/quotations.json';

// 创建必要的目录
async function ensureDirectories() {
    try {
        await fs.mkdir('./assets', { recursive: true });
        await fs.mkdir(quotationsDir, { recursive: true });
        
        // 检查quotations.json文件是否存在
        try {
            await fs.access(jsonFilePath);
        } catch {
            // 如果文件不存在，创建空的数组
            await fs.writeFile(jsonFilePath, JSON.stringify([], null, 2));
            console.log('已创建 quotations.json 文件');
        }
    } catch (error) {
        console.error('创建目录失败:', error);
    }
}

// 配置multer存储
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, quotationsDir);
    },
    filename: function (req, file, cb) {
        // 这里我们先使用临时文件名，稍后根据实际数据重命名
        cb(null, 'temp_' + Date.now() + path.extname(file.originalname));
    }
});

const upload = multer({ 
    storage: storage,
    limits: { fileSize: 10 * 1024 * 1024 }, // 10MB限制
    fileFilter: function (req, file, cb) {
        // 只接受图片文件
        if (file.mimetype === 'image/jpeg' || file.mimetype === 'image/png') {
            cb(null, true);
        } else {
            cb(new Error('只支持JPG和PNG格式的图片'));
        }
    }
});

// 处理上传
app.post('/upload', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ 
                success: false, 
                message: '没有上传图片文件' 
            });
        }

        const { title, width, height, format, footer, messages } = req.body;
        
        // 验证必填字段
        if (!title || !width || !height || !messages) {
            // 删除临时文件
            await fs.unlink(req.file.path);
            return res.status(400).json({ 
                success: false, 
                message: '缺少必填字段' 
            });
        }

        // 解析messages数组
        let messagesArray;
        try {
            messagesArray = JSON.parse(messages);
        } catch (error) {
            messagesArray = [messages];
        }

        // 生成文件名
        const finalFilename = `${title}.${format}`;
        const finalPath = path.join(quotationsDir, finalFilename);

        // 重命名文件
        await fs.rename(req.file.path, finalPath);
        
        // 创建新的条目
        const newEntry = {
            width: parseInt(width),
            height: parseInt(height),
            title: title,
        };
        if (format && format.trim() === 'jpg') {
            newEntry.format = 'jpg';
        }
        newEntry.messages = messagesArray;

        // 添加可选的footer
        if (footer && footer.trim()) {
            newEntry.footer = footer.trim();
        }


        // 读取现有的quotations.json
        const jsonData = await fs.readFile(jsonFilePath, 'utf8');
        const quotations = JSON.parse(jsonData);

        // 在数组头部插入新条目
        quotations.unshift(newEntry);

        // 写回文件
        await fs.writeFile(jsonFilePath, JSON.stringify(quotations, null, 2));

        res.json({
            success: true,
            message: '上传成功',
            data: {
                filename: finalFilename,
                entry: newEntry
            }
        });

    } catch (error) {
        console.error('上传失败:', error);
        
        // 清理临时文件
        if (req.file && req.file.path) {
            try {
                await fs.unlink(req.file.path);
            } catch (unlinkError) {
                console.error('清理临时文件失败:', unlinkError);
            }
        }
        
        res.status(500).json({ 
            success: false, 
            message: error.message || '上传失败' 
        });
    }
});

// 获取quotations列表
app.get('/quotations', async (req, res) => {
    try {
        const jsonData = await fs.readFile(jsonFilePath, 'utf8');
        const quotations = JSON.parse(jsonData);
        res.json({ success: true, data: quotations });
    } catch (error) {
        res.status(500).json({ 
            success: false, 
            message: '读取数据失败' 
        });
    }
});

// 提供前端页面
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'upload.html'));
});

// 启动服务器
async function startServer() {
    await ensureDirectories();
    
    app.listen(port, () => {
        console.log(`服务器运行在 http://localhost:${port}`);
        console.log(`上传页面: http://localhost:${port}/`);
    });
}

// 错误处理中间件
app.use((err, req, res, next) => {
    if (err instanceof multer.MulterError) {
        if (err.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).json({ 
                success: false, 
                message: '文件大小超过10MB限制' 
            });
        }
    }
    console.error(err.stack);
    res.status(500).json({ 
        success: false, 
        message: '服务器内部错误' 
    });
});

startServer();