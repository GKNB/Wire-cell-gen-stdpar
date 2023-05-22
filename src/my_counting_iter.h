template<typename IndexType = size_t>
class my_counting_iterator
{
  public:
    typedef IndexType         value_type;
    typedef const IndexType&  reference;
    typedef const IndexType*  pointer;
    typedef typename std::make_signed<IndexType>::type  difference_type;
    typedef std::random_access_iterator_tag             iterator_category;

    my_counting_iterator(): m_idx(0) {}
    explicit my_counting_iterator(IndexType x): m_idx(x) {}

    value_type operator *() const { return m_idx; }
    
    my_counting_iterator& operator ++() { ++m_idx; return *this; }
    my_counting_iterator operator ++(int) { my_counting_iterator copy{m_idx}; ++m_idx; return copy; }

    my_counting_iterator& operator --() { --m_idx; return *this; }
    my_counting_iterator operator --(int) { my_counting_iterator copy(m_idx); --m_idx; return copy; }
    
    my_counting_iterator& operator +=(difference_type by) { m_idx+=by; return *this; }
    my_counting_iterator& operator -=(difference_type by) { m_idx-=by; return *this; }

    value_type operator [](const difference_type &i) const { return m_idx + i; }

    friend my_counting_iterator operator +(my_counting_iterator const& i, difference_type n) { return my_counting_iterator(i.m_idx + n); }
    friend my_counting_iterator operator +(difference_type n, my_counting_iterator const& i) { return my_counting_iterator(i.m_idx + n); }
    friend difference_type   operator -(my_counting_iterator const& x, my_counting_iterator const& y) { return x.m_idx - y.m_idx; }
    friend my_counting_iterator operator -(my_counting_iterator const& i, difference_type n) { return my_counting_iterator(i.m_idx - n); }
  
    friend bool operator ==(my_counting_iterator const& lhs, my_counting_iterator const& rhs) { return lhs.m_idx == rhs.m_idx; }
    friend bool operator !=(my_counting_iterator const& lhs, my_counting_iterator const& rhs) { return !(lhs == rhs); }

    friend bool operator < (my_counting_iterator const& lhs, my_counting_iterator const& rhs) { return lhs.m_idx < rhs.m_idx; }
    friend bool operator > (my_counting_iterator const& lhs, my_counting_iterator const& rhs) { return rhs < lhs; }
    friend bool operator <=(my_counting_iterator const& lhs, my_counting_iterator const& rhs) { return !(lhs > rhs); }
    friend bool operator >=(my_counting_iterator const& lhs, my_counting_iterator const& rhs) { return !(lhs < rhs); }

  private:
    IndexType m_idx; // exposition
};

