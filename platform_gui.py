"""
一站式商家评论文本分析与智能回复平台（离线GUI）
  - 选项卡1: 评论与智能回复（列表+详情+选择/编辑/重新生成）
  - 选项卡2: 数据可视化（客户类型/同行情况/关注度方面/各星级分布）
运行: python platform_gui.py
"""
import os, sys, threading, json, tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, simpledialog
from PIL import Image, ImageTk
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aspect_reply_model as arm

DEV_PATH = 'data/dev.csv'
VIZ_DIR = 'visualization'
MERCHANTS_FILE = 'merchants.json'

VIZ_IMAGES = [
    ('customer_type_pie.png', '客户类型分布'),
    ('companion_bar.png', '同行情况'),
    ('aspect_bar.png', '关注度方面'),
    ('star_distribution.png', '各星级分布评分情况'),
]


def load_merchants():
    """加载商家列表：默认3个+用户自定义"""
    merchants = list(arm.MERCHANTS)
    if os.path.exists(MERCHANTS_FILE):
        try:
            with open(MERCHANTS_FILE, 'r', encoding='utf-8') as f:
                custom = json.load(f)
                if isinstance(custom, list):
                    merchants.extend(custom)
        except Exception:
            pass
    return merchants


def save_custom_merchant(merchant):
    """保存用户添加的商家到JSON"""
    custom = []
    if os.path.exists(MERCHANTS_FILE):
        try:
            with open(MERCHANTS_FILE, 'r', encoding='utf-8') as f:
                custom = json.load(f)
                if not isinstance(custom, list):
                    custom = []
        except Exception:
            custom = []
    custom.append(merchant)
    with open(MERCHANTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(custom, f, ensure_ascii=False, indent=2)


class PlatformApp:
    def __init__(self, root):
        self.root = root
        self.root.title("一站式商家评论文本分析与智能回复平台")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 750)

        self.df = None
        self.merchants = load_merchants()
        self.merchant = self.merchants[0] if self.merchants else arm.MERCHANTS[0]
        self.replies_cache = {}
        self.current_idx = None
        self.star_images = self._make_star_images()

        self._build_toolbar()
        self._build_notebook()
        self._load_data()

    def _build_toolbar(self):
        toolbar = ttk.Frame(self.root, padding=(8, 6))
        toolbar.pack(fill='x', side='top')

        ttk.Label(toolbar, text="商家:").pack(side='left', padx=(0, 4))
        self.merchant_var = tk.StringVar(value=self.merchant['name'])
        self.merchant_cb = ttk.Combobox(toolbar, textvariable=self.merchant_var,
                                        values=[m['name'] for m in self.merchants],
                                        state='readonly', width=22)
        self.merchant_cb.pack(side='left', padx=(0, 4))
        self.merchant_cb.bind('<<ComboboxSelected>>', self._on_merchant_change)

        ttk.Button(toolbar, text="添加商家", command=self._add_merchant_dialog).pack(side='left', padx=4)
        ttk.Button(toolbar, text="重新加载评论", command=self._load_data).pack(side='left', padx=4)

        ttk.Label(toolbar, text="ID 查询:").pack(side='left', padx=(16, 4))
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(toolbar, textvariable=self.search_var, width=14)
        search_entry.pack(side='left', padx=(0, 4))
        search_entry.bind('<Return>', lambda e: self._search_by_id())
        ttk.Button(toolbar, text="查询", command=self._search_by_id).pack(side='left', padx=2)
        ttk.Button(toolbar, text="清除", command=self._clear_search).pack(side='left', padx=2)

        self.status_var = tk.StringVar(value="尚未加载评论")
        ttk.Label(toolbar, textvariable=self.status_var, foreground='#555').pack(side='left', padx=12)

        api_status = "API: 已配置" if arm.CONFIG['api_key'] else "API: 未配置(将用本地模板)"
        api_color = '#1B7F3B' if arm.CONFIG['api_key'] else '#B0413E'
        ttk.Label(toolbar, text=api_status, foreground=api_color).pack(side='right')

    def _build_notebook(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)

        self.tab_reply = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_reply, text="  评论与智能回复  ")
        self._build_reply_tab()

        self.tab_viz = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_viz, text="  数据可视化  ")
        self._build_viz_tab()

    def _build_reply_tab(self):
        paned = ttk.PanedWindow(self.tab_reply, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=8, pady=8)

        # 左侧：评论列表
        left = ttk.Frame(paned)
        paned.add(left, weight=1)

        ttk.Label(left, text="评论列表", font=('微软雅黑', 11, 'bold')).pack(anchor='w', pady=(0, 4))

        # 快速浏览滑槽
        nav_frame = ttk.Frame(left)
        nav_frame.pack(fill='x', pady=(0, 4))
        ttk.Label(nav_frame, text="快速浏览:", font=('微软雅黑', 9)).pack(side='left')
        self.nav_scale = ttk.Scale(nav_frame, from_=0, to=100, orient='horizontal',
                                  command=self._on_nav_scale_change)
        self.nav_scale.pack(side='left', fill='x', expand=True, padx=6)
        self.nav_info_var = tk.StringVar(value="0 / 0")
        ttk.Label(nav_frame, textvariable=self.nav_info_var,
                  font=('微软雅黑', 9), width=14, anchor='e').pack(side='left', padx=(4, 0))
        self.nav_scale.bind('<Double-Button-1>', lambda e: self._nav_to_position(0))

        # Treeview + 滚动条
        tree_frame = ttk.Frame(left)
        tree_frame.pack(fill='both', expand=True)
        cols = ('idx', 'id', 'star', 'preview')
        self.tree = ttk.Treeview(tree_frame, columns=cols, show='headings', height=25)
        self.tree.heading('idx', text='#')
        self.tree.heading('id', text='ID')
        self.tree.heading('star', text='评分')
        self.tree.heading('preview', text='评论预览')
        self.tree.column('idx', width=40, anchor='center')
        self.tree.column('id', width=80, anchor='center')
        self.tree.column('star', width=60, anchor='center')
        self.tree.column('preview', width=380, anchor='w')
        self.tree.bind('<<TreeviewSelect>>', self._on_select_review)
        self.tree.bind('<MouseWheel>',
                       lambda e: self.tree.yview_scroll(int(-1 * (e.delta / 120)), 'units'))

        vsb = ttk.Scrollbar(tree_frame, orient='vertical', command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient='horizontal', command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        # 右侧：详情区
        right = ttk.Frame(paned)
        paned.add(right, weight=2)

        top_frame = ttk.Frame(right)
        top_frame.pack(fill='x', pady=(0, 6))
        ttk.Label(top_frame, text="评论 ID: ", font=('微软雅黑', 10, 'bold')).pack(side='left')
        self.id_var = tk.StringVar(value='-')
        ttk.Label(top_frame, textvariable=self.id_var, font=('微软雅黑', 10)).pack(side='left', padx=(0, 16))
        ttk.Label(top_frame, text="星级: ", font=('微软雅黑', 10, 'bold')).pack(side='left')
        self.star_label = ttk.Label(top_frame, image='', compound='left')
        self.star_label.pack(side='left')

        ttk.Label(right, text="评论内容:", font=('微软雅黑', 10, 'bold')).pack(anchor='w', pady=(0, 2))
        self.review_text = scrolledtext.ScrolledText(right, height=8, wrap='word',
                                                      font=('微软雅黑', 10), bg='#FAFAFA')
        self.review_text.pack(fill='x', pady=(0, 8))
        self.review_text.configure(state='disabled')

        ttk.Label(right, text="智能回复 (可编辑):", font=('微软雅黑', 10, 'bold')).pack(anchor='w', pady=(0, 2))
        self.reply_text = scrolledtext.ScrolledText(right, height=10, wrap='word',
                                                     font=('微软雅黑', 10), bg='#FFFCEB')
        self.reply_text.pack(fill='both', expand=True, pady=(0, 8))

        btn_frame = ttk.Frame(right)
        btn_frame.pack(fill='x', pady=(4, 0))
        ttk.Button(btn_frame, text="选择回复 (复制到剪贴板)",
                   command=self._select_reply).pack(side='left', padx=(0, 8))
        ttk.Button(btn_frame, text="编辑文本",
                   command=self._edit_reply).pack(side='left', padx=(0, 8))
        ttk.Button(btn_frame, text="重新生成",
                   command=self._regenerate_reply).pack(side='left', padx=(0, 8))
        self.reply_source_var = tk.StringVar(value="")
        ttk.Label(btn_frame, textvariable=self.reply_source_var, foreground='#888').pack(side='right')

    def _build_viz_tab(self):
        container = ttk.Frame(self.tab_viz)
        container.pack(fill='both', expand=True, padx=8, pady=8)

        ttk.Label(container, text="数据可视化分析",
                  font=('微软雅黑', 13, 'bold')).pack(anchor='w', pady=(0, 6))

        # Canvas + Scrollbar 实现可滚动区域
        canvas = tk.Canvas(container, highlightthickness=0, bg='#F5F5F5')
        vsb = ttk.Scrollbar(container, orient='vertical', command=canvas.yview)
        hsb = ttk.Scrollbar(container, orient='horizontal', command=canvas.xview)
        canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side='right', fill='y')
        hsb.pack(side='bottom', fill='x')
        canvas.pack(side='left', fill='both', expand=True)

        scroll_frame = ttk.Frame(canvas)
        scroll_window = canvas.create_window((0, 0), window=scroll_frame, anchor='nw')

        def _on_canvas_resize(event):
            canvas.itemconfig(scroll_window, width=event.width)
        canvas.bind('<Configure>', _on_canvas_resize)

        def _on_wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        canvas.bind('<Enter>', lambda e: canvas.bind_all('<MouseWheel>', _on_wheel))
        canvas.bind('<Leave>', lambda e: canvas.unbind_all('<MouseWheel>'))

        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox('all'))
        scroll_frame.bind('<Configure>', _on_frame_configure)

        # 2x2 网格
        self.viz_images_refs = []
        for i, (fname, title) in enumerate(VIZ_IMAGES):
            row, col = i // 2, i % 2
            cell = ttk.LabelFrame(scroll_frame, text=title, padding=6)
            cell.grid(row=row, column=col, sticky='nsew', padx=8, pady=8)
            scroll_frame.rowconfigure(row, weight=1)
            scroll_frame.columnconfigure(col, weight=1)

            fpath = os.path.join(VIZ_DIR, fname)
            if not os.path.exists(fpath):
                ttk.Label(cell, text=f"[图片缺失]\n{fname}", foreground='red').pack(expand=True)
                continue

            try:
                img = Image.open(fpath)
                max_w = 620
                if img.width > max_w:
                    ratio = max_w / img.width
                    img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                lbl = ttk.Label(cell, image=photo)
                lbl.image = photo
                lbl.pack(fill='both', expand=True)
                self.viz_images_refs.append(photo)
            except Exception as e:
                ttk.Label(cell, text=f"[加载失败]\n{e}", foreground='red').pack(expand=True)

        ttk.Label(scroll_frame,
                  text="图片来源于 visualization/ 目录下的历史分析结果。(可上下/左右滚动)",
                  foreground='#777').grid(row=len(VIZ_IMAGES) // 2 + 1, column=0,
                                           columnspan=2, sticky='w', pady=(8, 0))

    def _make_star_images(self):
        """生成1-5星的彩色图片"""
        imgs = {}
        try:
            from PIL import ImageDraw, ImageFont
            for n in range(1, 6):
                img = Image.new('RGBA', (110, 22), (255, 255, 255, 0))
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("C:\\Windows\\Fonts\\msyhbd.ttc", 18)
                except Exception:
                    font = ImageFont.load_default()
                draw.text((0, 1), '★' * n + '☆' * (5 - n), font=font, fill=(245, 180, 60))
                imgs[n] = img
        except Exception:
            pass
        return imgs

    def _render_stars(self, n):
        if not self.star_images or n not in self.star_images:
            return ''
        return ImageTk.PhotoImage(self.star_images[int(n)])

    def _load_data(self):
        """加载dev.csv评论"""
        if not os.path.exists(DEV_PATH):
            self.status_var.set(f"未找到 {DEV_PATH}")
            messagebox.showerror("错误", f"未找到评论文件: {DEV_PATH}")
            return
        try:
            self.df = pd.read_csv(DEV_PATH, encoding='utf-8-sig')
            self.tree.delete(*self.tree.get_children())
            for idx, row in self.df.iterrows():
                rid = str(row.get('id', ''))
                star = row.get('star', 0)
                try:
                    star_int = int(round(float(star)))
                except Exception:
                    star_int = 0
                preview = str(row.get('review', '') or '')[:30].replace('\n', ' ')
                self.tree.insert('', 'end', iid=str(idx),
                                 values=(idx + 1, rid, '★' * star_int, preview))
            self.status_var.set(f"已加载 {len(self.df)} 条评论 | 商家: {self.merchant['name']}")
            self.replies_cache = {}
            self.nav_scale.set(0)
            self.nav_info_var.set(f"1 / {len(self.df)}")
            self.tree.bind('<<TreeviewMotion>>', self._sync_scale_from_treeview)
        except Exception as e:
            messagebox.showerror("加载失败", str(e))

    def _on_merchant_change(self, event):
        name = self.merchant_var.get()
        for m in self.merchants:
            if m['name'] == name:
                self.merchant = m
                break
        self.replies_cache = {}
        self.status_var.set(f"商家已切换: {self.merchant['name']} | 缓存已清空")

    def _on_nav_scale_change(self, val):
        if self.df is None or len(self.df) == 0:
            return
        total = len(self.df)
        idx = int(float(val) / 100 * max(total - 1, 0))
        self.nav_info_var.set(f"{idx + 1} / {total}")
        self.tree.yview_moveto(idx / max(total - 1, 1))

    def _nav_to_position(self, idx):
        if self.df is None or len(self.df) == 0:
            return
        total = len(self.df)
        idx = max(0, min(idx, total - 1))
        self.nav_scale.set(idx / max(total - 1, 1) * 100)
        self.nav_info_var.set(f"{idx + 1} / {total}")
        self.tree.yview_moveto(idx / max(total - 1, 1))

    def _sync_scale_from_treeview(self, event=None):
        if self.df is None or len(self.df) == 0:
            return
        try:
            top_frac = self.tree.yview()[0]
            total = len(self.df)
            idx = int(top_frac * (total - 1))
            idx = max(0, min(idx, total - 1))
            current = float(self.nav_scale.get())
            target = idx / max(total - 1, 1) * 100
            if abs(current - target) > 1:
                self.nav_scale.set(target)
            self.nav_info_var.set(f"{idx + 1} / {total}")
        except Exception:
            pass

    def _add_merchant_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("添加商家")
        dialog.geometry("440x420")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        ttk.Label(dialog, text="填写商家信息",
                  font=('微软雅黑', 11, 'bold')).pack(anchor='w', padx=12, pady=(12, 6))

        form = ttk.Frame(dialog, padding=12)
        form.pack(fill='both', expand=True)

        fields = [
            ('商家名称 *', 'name', ''),
            ('菜系 (如川菜/烧烤)', 'cuisine', ''),
            ('招牌菜 (逗号分隔)', 'signature', ''),
            ('口碑评分 (1-5)', 'rating', '4.0'),
            ('地址', 'address', ''),
            ('人均 (元)', 'price', ''),
        ]
        entries = {}
        for i, (label, key, default) in enumerate(fields):
            ttk.Label(form, text=label).grid(row=i, column=0, sticky='w', padx=4, pady=6)
            var = tk.StringVar(value=default)
            ttk.Entry(form, textvariable=var, width=28).grid(row=i, column=1, padx=4, pady=6)
            entries[key] = var

        def save():
            name = entries['name'].get().strip()
            if not name:
                messagebox.showerror("错误", "请输入商家名称", parent=dialog)
                return
            try:
                rating = float(entries['rating'].get() or '4.0')
                if not (1.0 <= rating <= 5.0):
                    raise ValueError()
            except ValueError:
                messagebox.showerror("错误", "口碑评分需为1-5的数字", parent=dialog)
                return

            merchant = {
                'name': name, 'cuisine': entries['cuisine'].get().strip(),
                'signature': [s.strip() for s in entries['signature'].get().split(',') if s.strip()],
                'rating': rating, 'address': entries['address'].get().strip(),
                'price': entries['price'].get().strip(),
            }
            for m in self.merchants:
                if m['name'] == name:
                    messagebox.showerror("错误", f"商家 '{name}' 已存在", parent=dialog)
                    return

            save_custom_merchant(merchant)
            self.merchants = load_merchants()
            self.merchant_cb['values'] = [m['name'] for m in self.merchants]
            self.merchant_var.set(name)
            for m in self.merchants:
                if m['name'] == name:
                    self.merchant = m
                    break
            self.replies_cache = {}
            messagebox.showinfo("成功", f"商家 '{name}' 已添加并保存", parent=dialog)
            self.status_var.set(f"新增商家: {name} | 已保存到 {MERCHANTS_FILE}")
            dialog.destroy()

        btn_frame = ttk.Frame(dialog, padding=8)
        btn_frame.pack(side='bottom', fill='x')
        ttk.Button(btn_frame, text="保存", command=save).pack(side='left', padx=4)
        ttk.Button(btn_frame, text="取消", command=dialog.destroy).pack(side='left', padx=4)

    def _search_by_id(self):
        if self.df is None:
            messagebox.showinfo("提示", "请先加载评论数据")
            return
        query = self.search_var.get().strip()
        if not query:
            self._load_data()
            return
        self.tree.delete(*self.tree.get_children())
        found = 0
        for idx, row in self.df.iterrows():
            rid = str(row.get('id', ''))
            if query in rid:
                star = row.get('star', 0)
                try:
                    star_int = int(round(float(star)))
                except Exception:
                    star_int = 0
                preview = str(row.get('review', '') or '')[:30].replace('\n', ' ')
                self.tree.insert('', 'end', iid=str(idx),
                                 values=(idx + 1, rid, '★' * star_int, preview))
                found += 1
        if found == 0:
            messagebox.showinfo("查询结果", f"未找到ID包含 '{query}' 的评论")
            self.status_var.set(f"ID查询: {query} | 未找到匹配")
        else:
            self.status_var.set(f"ID查询: {query} | 找到 {found} 条")
            first = self.tree.get_children()
            if first:
                self.tree.selection_set(first[0])
                self.tree.focus(first[0])
                self._on_select_review(None)

    def _clear_search(self):
        self.search_var.set('')
        self._load_data()

    def _on_select_review(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        self.current_idx = idx
        row = self.df.iloc[idx]
        rid = str(row.get('id', ''))
        star = row.get('star', 0)
        try:
            star_int = int(round(float(star)))
        except Exception:
            star_int = 0
        review_text = str(row.get('review', '') or '')

        self.id_var.set(rid)
        if self.star_images and star_int in self.star_images:
            photo = ImageTk.PhotoImage(self.star_images[star_int])
            self.star_label.config(image=photo)
            self.star_label.image = photo
        else:
            self.star_label.config(image='')

        self.review_text.config(state='normal')
        self.review_text.delete('1.0', 'end')
        self.review_text.insert('1.0', review_text)
        self.review_text.config(state='disabled')

        self.reply_text.delete('1.0', 'end')
        if idx in self.replies_cache:
            self.reply_text.insert('1.0', self.replies_cache[idx]['reply'])
            self.reply_source_var.set(f"来源: {self.replies_cache[idx]['source']}")
        else:
            self.reply_source_var.set("正在生成回复...")
            self._async_generate_reply(idx, review_text)

    def _async_generate_reply(self, idx, review_text):
        def worker():
            try:
                result = arm.generate_reply(review_text, self.merchant, use_api=True)
                self.root.after(0, lambda: self._update_reply(idx, result))
            except Exception as e:
                self.root.after(0, lambda: self._update_reply(idx, {'reply': f'[生成失败] {e}',
                                                                     'source': '错误'}))
        threading.Thread(target=worker, daemon=True).start()

    def _update_reply(self, idx, result):
        if self.current_idx != idx:
            return
        self.replies_cache[idx] = result
        self.reply_text.delete('1.0', 'end')
        self.reply_text.insert('1.0', result['reply'])
        self.reply_source_var.set(f"来源: {result['source']}")

    def _select_reply(self):
        content = self.reply_text.get('1.0', 'end-1c').strip()
        if not content:
            messagebox.showinfo("提示", "智能回复为空, 请先选择评论或重新生成")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        messagebox.showinfo("已复制", "智能回复已复制到剪贴板, 可直接粘贴使用")

    def _edit_reply(self):
        self.reply_text.focus_set()
        self.reply_source_var.set("编辑模式 (直接在文本框内修改)")

    def _regenerate_reply(self):
        if self.current_idx is None:
            messagebox.showinfo("提示", "请先在左侧列表选择一条评论")
            return
        idx = self.current_idx
        review_text = str(self.df.iloc[idx].get('review', '') or '')
        self.reply_text.delete('1.0', 'end')
        self.reply_source_var.set("正在重新生成回复...")
        if idx in self.replies_cache:
            del self.replies_cache[idx]
        self._async_generate_reply(idx, review_text)


def main():
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except Exception:
        pass
    try:
        style.configure('.', font=('微软雅黑', 10))
    except Exception:
        pass
    PlatformApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()